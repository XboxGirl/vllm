# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Callable

import torch

from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.compressed_tensors.schemes.compressed_tensors_scheme import (  # noqa: E501
    CompressedTensorsScheme,
)
from vllm.model_executor.layers.quantization.utils.marlin_utils import (
    marlin_repeat_scales_on_all_ranks,
)
from vllm.model_executor.parameter import (
    BasevLLMParameter,
    ChannelQuantScaleParameter,
    GroupQuantScaleParameter,
    PackedvLLMParameter,
)

__all__ = ["CompressedTensorsW4A16XPU"]

logger = init_logger(__name__)


class CompressedTensorsW4A16XPU(CompressedTensorsScheme):
    """XPU-specific W4A16 path backed by oneDNN int4 GEMM."""

    def __init__(self, strategy: str, group_size: int | None = None):
        self.strategy = strategy
        self.group_size = -1 if group_size is None else group_size
        self.num_bits = 4
        self.pack_factor = 32 // self.num_bits

        if self.group_size == -1 and self.strategy != "channel":
            raise ValueError(
                "XPU W4A16 requires group quantization or channelwise "
                "quantization."
            )

    @classmethod
    def get_min_capability(cls) -> int:
        # This XPU-only scheme should not be gated by CUDA capability checks.
        return 0

    def create_weights(
        self,
        layer: torch.nn.Module,
        output_size: int,
        input_size: int,
        output_partition_sizes: list[int],
        input_size_per_partition: int,
        params_dtype: torch.dtype,
        weight_loader: Callable,
        **kwargs,
    ):
        output_size_per_partition = sum(output_partition_sizes)
        layer.xpu_output_size = output_size_per_partition

        # If group_size is -1, we are in channelwise case.
        group_size = self.group_size if self.group_size != -1 else input_size
        row_parallel = input_size != input_size_per_partition
        partition_scales = not marlin_repeat_scales_on_all_ranks(
            False, self.group_size, row_parallel
        )

        scales_and_zp_size = input_size // group_size

        if partition_scales:
            assert input_size_per_partition % group_size == 0
            scales_and_zp_size = input_size_per_partition // group_size

        weight = PackedvLLMParameter(
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
            packed_factor=self.pack_factor,
            packed_dim=1,
            data=torch.empty(
                output_size_per_partition,
                input_size_per_partition // self.pack_factor,
                dtype=torch.int32,
            ),
        )

        weight_scale_args = {
            "weight_loader": weight_loader,
            "data": torch.empty(
                output_size_per_partition,
                scales_and_zp_size,
                dtype=params_dtype,
            ),
        }

        if not partition_scales:
            weight_scale = ChannelQuantScaleParameter(
                output_dim=0, **weight_scale_args
            )
        else:
            weight_scale = GroupQuantScaleParameter(
                output_dim=0, input_dim=1, **weight_scale_args
            )

        # A 2D array defining the original shape of the weights
        # before packing
        weight_shape = BasevLLMParameter(
            data=torch.empty(2, dtype=torch.int64), weight_loader=weight_loader
        )

        layer.register_parameter("weight_packed", weight)
        layer.register_parameter("weight_scale", weight_scale)
        layer.register_parameter("weight_shape", weight_shape)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        missing_attrs = [
            name
            for name in ("weight_packed", "weight_scale")
            if not hasattr(layer, name)
        ]
        if missing_attrs:
            raise AttributeError(
                "CompressedTensorsW4A16XPU expected loaded attributes "
                f"{missing_attrs} on layer {layer.__class__.__name__}. "
                "This usually means create_weights naming diverged from the "
                "checkpoint loader mapping."
            )

        qw = layer.weight_packed.data  # (out, in//8), int32 — compressed-tensors layout
        scales = layer.weight_scale.data  # (out, num_groups) — compressed-tensors layout
        device = qw.device

        out_size = qw.shape[0]
        in_size = qw.shape[1] * self.pack_factor

        if out_size % self.pack_factor != 0:
            raise ValueError(
                f"output_size ({out_size}) must be divisible by pack_factor "
                f"({self.pack_factor}) for XPU int4 GEMM repacking."
            )

        # --- Repack (out, in//8) → (in, out//8) ---
        # Compressed-tensors packs along the input dim; the oneDNN kernel (like
        # AWQ) expects rows=input, columns=packed-output.  Full unpack →
        # transpose → repack is required because the pack dimension changes.
        shifts = torch.arange(
            0, 32, self.num_bits, dtype=torch.int32, device=device
        )
        mask = (1 << self.num_bits) - 1

        # Unpack: (out, in//8) → (out, in)
        unpacked = ((qw.unsqueeze(-1) >> shifts) & mask).reshape(out_size, in_size)

        # Transpose: (out, in) → (in, out)
        unpacked = unpacked.T.contiguous()

        # Repack along output dim: (in, out//8)
        repacked = (
            unpacked
            .reshape(in_size, out_size // self.pack_factor, self.pack_factor)
            .to(torch.int32)
            .__lshift__(shifts[None, None, :])
            .sum(dim=-1, dtype=torch.int32)
        )

        layer.qweight = torch.nn.Parameter(repacked, requires_grad=False)

        # Re-derive output size from the repacked weight for safety.
        layer.xpu_output_size = out_size

        # --- Transpose scales (out, num_groups) → (num_groups, out) ---
        # AWQ/oneDNN kernel convention is (groups, out).
        layer.scales = torch.nn.Parameter(
            scales.T.contiguous(), requires_grad=False
        )

        # Synthesised scalar zero-point placeholder for symmetric quant.
        layer.qzeros = torch.nn.Parameter(
            torch.zeros(1, dtype=torch.int8, device=device),
            requires_grad=False,
        )

        try:
            from vllm_xpu_kernels.quantization._quantize_convert import (
                transpose_onednn_woq_format,
            )
        except ImportError as exc:
            raise RuntimeError(
                "Failed to import XPU compressed-tensors conversion kernels. "
                "Ensure vllm_xpu_kernels is installed and importable."
            ) from exc

        # GPTQ symmetric format: oneDNN broadcasts synthesized scalar zero-point.
        try:
            transpose_onednn_woq_format(layer, "gptq", is_sym=True)
        except Exception as exc:
            logger.exception(
                "W4A16 XPU transpose failed for %s "
                "(qweight=%s, scales=%s, group_size=%s).",
                layer.__class__.__name__,
                tuple(layer.qweight.shape),
                tuple(layer.scales.shape),
                self.group_size,
            )
            raise RuntimeError(
                "Failed to prepare compressed-tensors W4A16 weights for XPU "
                "int4 GEMM."
            ) from exc

        logger.info_once(
            "Using CompressedTensorsW4A16XPU oneDNN int4 GEMM path "
            "(symmetric gptq layout conversion enabled)."
        )

        del layer.weight_packed
        del layer.weight_scale

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        reshaped_x = x.reshape(-1, x.shape[-1])

        if self.group_size != -1:
            group_size = self.group_size
        else:
            group_size = reshaped_x.shape[-1]

        out = torch.ops._xpu_C.int4_gemm_w4a16(
            reshaped_x,
            layer.qweight,
            bias,
            layer.scales,
            layer.qzeros,
            group_size,
            None,
        )
        out_shape = x.shape[:-1] + (layer.xpu_output_size,)
        return out.reshape(out_shape)


