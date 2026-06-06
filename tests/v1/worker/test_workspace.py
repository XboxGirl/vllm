# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.v1.worker.workspace import (
    current_workspace_manager,
    init_workspace_manager,
    lock_workspace,
    reset_workspace_manager,
)


def test_can_get_simultaneous_reports_locked_capacity() -> None:
    reset_workspace_manager()
    try:
        init_workspace_manager(torch.device("cpu"))
        manager = current_workspace_manager()

        shape = ((4,), torch.float32)
        larger_shape = ((1024 * 1024,), torch.float32)

        assert manager.can_get_simultaneous(shape)
        manager.get_simultaneous(shape)
        lock_workspace()

        assert manager.can_get_simultaneous(shape)
        assert not manager.can_get_simultaneous(larger_shape)

        try:
            manager.get_simultaneous(larger_shape)
        except AssertionError as exc:
            assert "Workspace is locked" in str(exc)
        else:
            raise AssertionError("locked workspace unexpectedly grew")
    finally:
        reset_workspace_manager()


def test_workspace_lock_allows_padded_near_boundary_request() -> None:
    reset_workspace_manager()
    try:
        init_workspace_manager(torch.device("cpu"))
        manager = current_workspace_manager()

        one_mib_floats = 256 * 1024
        warmup_shape = ((one_mib_floats,), torch.float32)
        slightly_larger_shape = ((one_mib_floats + 1,), torch.float32)
        too_large_shape = ((2 * one_mib_floats + 1,), torch.float32)

        manager.get_simultaneous(warmup_shape)
        lock_workspace()

        assert manager.can_get_simultaneous(slightly_larger_shape)
        manager.get_simultaneous(slightly_larger_shape)
        assert not manager.can_get_simultaneous(too_large_shape)
    finally:
        reset_workspace_manager()
