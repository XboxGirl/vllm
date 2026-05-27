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

        qw = layer.weight_packed.data  # (out, in//8), int32, packed along input dim
        scales = layer.weight_scale.data  # (out, num_groups)

        logger.debug(
            "CompressedTensorsW4A16XPU pre-conversion shapes for %s: "
            "weight_packed=%s weight_scale=%s group_size=%s",
            layer.__class__.__name__,
            tuple(qw.shape),
            tuple(scales.shape),
            self.group_size,
        )

        # int4_gemm_w4a16 fake/real paths produce output width from
        # qweight.shape[1], so qweight must have output dim in axis 1.
        # Convert compressed-tensors (out, in//8) -> (in//8, out).
        layer.qweight = torch.nn.Parameter(
            qw.t().contiguous(), requires_grad=False
        )

        # Scales: (out, num_groups) → (num_groups, out) — kernel convention.
        layer.scales = torch.nn.Parameter(
            scales.t().contiguous(), requires_grad=False
        )

        # out_size is axis 0 of qw (before transpose).
        layer.xpu_output_size = qw.shape[0]

        # Synthesised scalar zero-point placeholder for symmetric quant.
        layer.qzeros = torch.nn.Parameter(
            torch.zeros(1, dtype=torch.int8, device=qw.device),
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

        logger.debug(
            "CompressedTensorsW4A16XPU post-conversion shapes for %s: "
            "qweight=%s scales=%s qzeros=%s xpu_output_size=%s",
            layer.__class__.__name__,
            tuple(layer.qweight.shape),
            tuple(layer.scales.shape),
            tuple(layer.qzeros.shape),
            layer.xpu_output_size,
        )

        logger.info_once(
            "Using CompressedTensorsW4A16XPU oneDNN int4 GEMM path "
            "(symmetric gptq layout conversion enabled)."
        )
        logger.info_once(
            "CompressedTensorsW4A16XPU prepared shapes: qweight=%s scales=%s "
            "qzeros=%s xpu_output_size=%s",
            tuple(layer.qweight.shape),
            tuple(layer.scales.shape),
            tuple(layer.qzeros.shape),
            layer.xpu_output_size,
        )

        # int4_gemm_w4a16 returns width = qweight.shape[1].
        # Keep this invariant explicit to fail early with clear diagnostics.
        if layer.qweight.shape[1] != layer.xpu_output_size:
            raise RuntimeError(
                "CompressedTensorsW4A16XPU produced an invalid qweight layout: "
                f"qweight.shape={tuple(layer.qweight.shape)} "
                f"xpu_output_size={layer.xpu_output_size}. "
                "Expected qweight.shape[1] == xpu_output_size after conversion."
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

        # Preferred convention: output channels are qweight axis 1.
        qweight_for_gemm = layer.qweight
        expected_out_width = getattr(layer, "xpu_output_size", None)

        logger.debug(
            "CompressedTensorsW4A16XPU apply for %s: input=%s reshaped=%s "
            "qweight=%s qweight_for_gemm=%s scales=%s qzeros=%s "
            "group_size=%s bias=%s",
            layer.__class__.__name__,
            tuple(x.shape),
            tuple(reshaped_x.shape),
            tuple(layer.qweight.shape),
            tuple(qweight_for_gemm.shape),
            tuple(layer.scales.shape),
            tuple(layer.qzeros.shape),
            group_size,
            None if bias is None else tuple(bias.shape),
        )

        out = torch.ops._xpu_C.int4_gemm_w4a16(
            reshaped_x,
            qweight_for_gemm,
            bias,
            layer.scales,
            layer.qzeros,
            group_size,
            None,
        )

        out_width = out.shape[-1]
        # Be defensive against backend layout mismatches: if kernel output width
        # disagrees with expected layer output, retry once with transposed
        # qweight and keep the version that matches expected width.
        if expected_out_width is not None and out_width != expected_out_width:
            alt_qweight = qweight_for_gemm.t().contiguous()
            logger.warning(
                "CompressedTensorsW4A16XPU got unexpected output width %s "
                "(expected %s) for %s; retrying with transposed qweight "
                "shape=%s.",
                out_width,
                expected_out_width,
                layer.__class__.__name__,
                tuple(alt_qweight.shape),
            )
            alt_out = torch.ops._xpu_C.int4_gemm_w4a16(
                reshaped_x,
                alt_qweight,
                bias,
                layer.scales,
                layer.qzeros,
                group_size,
                None,
            )
            if alt_out.shape[-1] == expected_out_width:
                out = alt_out
                out_width = alt_out.shape[-1]

        out_shape = x.shape[:-1] + (out_width,)
        return out.reshape(out_shape)


