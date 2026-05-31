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
        larger_shape = ((1024,), torch.float32)

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
