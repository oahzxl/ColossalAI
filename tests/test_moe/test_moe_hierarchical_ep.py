import os

import pytest
import torch
import torch.distributed as dist

import colossalai
from colossalai.moe import SparseMLP
from colossalai.moe.manager import MOE_MANAGER
from colossalai.moe.utils import sync_moe_model_param
from colossalai.tensor.moe_tensor.api import get_ep_group, get_ep_rank, get_ep_size, is_moe_tensor
from colossalai.testing import assert_equal_in_group, rerun_if_address_is_in_use, spawn
from colossalai.utils import get_current_device
from tests.test_moe.moe_utils import MoeGradientHandler


def sync_tp_from_local(tp_model: SparseMLP, local_model: SparseMLP, assert_grad_flag: bool = False) -> None:
    """Sync the parameters of tp model from local model

    Args:
        tp_model (MoeModule)
        local_model (MoeModule)
    """
    for (tp_name, tp_param), (local_name, local_param) in zip(
        tp_model.named_parameters(), local_model.named_parameters()
    ):
        assert tp_name == local_name
        if not is_moe_tensor(tp_param):
            if assert_grad_flag:
                assert torch.allclose(tp_param, local_param)
                assert torch.allclose(tp_param.grad, local_param.grad)
            else:
                tp_param.data.copy_(local_param.data)
            continue

        tp_rank = get_ep_rank(tp_param)
        tp_dim = [i for i, (d1, d2) in enumerate(zip(tp_param.shape, local_param.shape)) if d1 != d2][0]
        tp_slice = [slice(None)] * tp_dim + [
            slice(tp_param.shape[tp_dim] * tp_rank, tp_param.shape[tp_dim] * (tp_rank + 1))
        ]

        if assert_grad_flag:
            assert torch.allclose(tp_param, local_param[tuple(tp_slice)])
            assert torch.allclose(tp_param.grad, local_param.grad[tuple(tp_slice)])
        else:
            tp_param.data.copy_(local_param[tuple(tp_slice)].data)


def sync_tp_from_ep(tp_model: SparseMLP, ep_model: SparseMLP, assert_grad_flag: bool = False) -> None:
    """Sync the parameters of tp model from ep model

    Args:
        tp_model (MoeModule)
        ep_model (MoeModule)
    """
    for (tp_name, tp_param), (ep_name, ep_param) in zip(tp_model.named_parameters(), ep_model.named_parameters()):
        assert tp_name == ep_name
        if not is_moe_tensor(tp_param):
            if assert_grad_flag:
                assert torch.allclose(tp_param, ep_param)
                assert torch.allclose(tp_param.grad, ep_param.grad)
            else:
                tp_param.data.copy_(ep_param.data)
            continue

        # gather param from ep model
        param_list = [torch.zeros_like(ep_param) for _ in range(get_ep_size(ep_param))]
        dist.all_gather(param_list, ep_param, group=get_ep_group(ep_param))
        all_param = torch.cat(param_list, dim=0)
        if assert_grad_flag:
            grad_list = [torch.zeros_like(ep_param) for _ in range(get_ep_size(ep_param))]
            dist.all_gather(grad_list, ep_param.grad, group=get_ep_group(ep_param))
            all_grad = torch.cat(grad_list, dim=0)

        # get tp param
        tp_dim = [i for i, (d1, d2) in enumerate(zip(tp_param.shape[1:], all_param.shape[1:])) if d1 != d2][0] + 1
        tp_rank = get_ep_rank(tp_param)
        tp_slice = [slice(None)] * tp_dim + [
            slice(tp_param.shape[tp_dim] * tp_rank, tp_param.shape[tp_dim] * (tp_rank + 1))
        ]
        new_tp_param = all_param[tuple(tp_slice)]
        if assert_grad_flag:
            new_grad = all_grad[tuple(tp_slice)]
        if assert_grad_flag:
            assert torch.allclose(tp_param, new_tp_param)
            assert torch.allclose(tp_param.grad, new_grad)
        else:
            tp_param.data.copy_(new_tp_param.data)


def sync_local_from_ep(local_model: SparseMLP, ep_model: SparseMLP, assert_grad_flag: bool = False) -> None:
    """Sync the parameters of tp model from ep model

    Args:
        local_model (MoeModule)
        ep_model (MoeModule)
    """
    for (local_name, local_param), (ep_name, ep_param) in zip(
        local_model.named_parameters(), ep_model.named_parameters()
    ):
        assert local_name == ep_name
        if "experts" not in local_name:
            if assert_grad_flag:
                assert torch.allclose(local_param, ep_param)
                assert torch.allclose(
                    local_param.grad, ep_param.grad
                ), f"name: {local_name}, local: {local_param.grad}, ep: {ep_param.grad}"
            else:
                local_param.data.copy_(ep_param.data)
            continue

        # gather param from ep model
        param_list = [torch.zeros_like(ep_param) for _ in range(get_ep_size(ep_param))]
        dist.all_gather(param_list, ep_param, group=get_ep_group(ep_param))
        all_param = torch.cat(param_list, dim=0)
        if assert_grad_flag:
            grad_list = [torch.zeros_like(ep_param) for _ in range(get_ep_size(ep_param))]
            dist.all_gather(grad_list, ep_param.grad, group=get_ep_group(ep_param))
            all_grad = torch.cat(grad_list, dim=0)

        if assert_grad_flag:
            assert torch.allclose(local_param, all_param)
            assert torch.allclose(local_param.grad, all_grad)
        else:
            local_param.data.copy_(all_param.data)


def run_test(rank: int, world_size: int, port: int, num_experts: int, batch_size: int, dim: int):
    assert batch_size % world_size == 0

    colossalai.launch(config=dict(), rank=rank, world_size=world_size, host="localhost", port=port, backend="nccl")

    MOE_MANAGER.__init__()
    MOE_MANAGER.setup(parallel=None)
    local_model = SparseMLP(num_experts=num_experts, hidden_size=dim, intermediate_size=dim * 2)
    MOE_MANAGER.__init__()
    MOE_MANAGER.setup(parallel="EP")
    # fake to be 2 nodes
    os.environ["LOCAL_WORLD_SIZE"] = str(world_size // 2)
    ep_model = SparseMLP(
        num_experts=num_experts, hidden_size=dim, intermediate_size=dim * 2, enable_hierarchical_comm=True
    )
    ep_model = ep_model.to(get_current_device())
    local_model = local_model.to(get_current_device())

    # sync ep param
    sync_moe_model_param(ep_model)
    dist_dict = MOE_MANAGER.parallel_info_dict
    assert_equal_in_group(ep_model.experts.wi.data, dist_dict[world_size].dp_group)
    assert_equal_in_group(ep_model.experts.wo.data, dist_dict[world_size].dp_group)
    ep_grad_handler = MoeGradientHandler(ep_model)
    # sync local param
    sync_local_from_ep(local_model, ep_model)

    rank = dist.get_rank()
    input_data = torch.randn(batch_size, dim, device=get_current_device())
    micro_batch_size = batch_size // world_size
    index = rank * micro_batch_size
    # NOTE: ep & tp takes in sharded data for each process
    shard_data = input_data.detach()[index : index + micro_batch_size]

    out_local = local_model(input_data)
    MOE_MANAGER.reset_loss()
    dist.barrier()
    out_ep = ep_model(shard_data)
    MOE_MANAGER.reset_loss()

    out_local_slice = out_local[index : index + micro_batch_size]
    assert torch.allclose(
        out_ep, out_local_slice, atol=1e-6
    ), f"Rank {rank} failed, max diff: {torch.max(torch.abs(out_ep - out_local_slice))}"

    out_local.mean().backward()
    out_ep.mean().backward()
    ep_grad_handler.handle_gradient()

    assert_equal_in_group(ep_model.experts.wi.grad, dist_dict[world_size].dp_group)
    assert_equal_in_group(ep_model.experts.wo.grad, dist_dict[world_size].dp_group)
    sync_local_from_ep(local_model, ep_model, assert_grad_flag=True)


@pytest.mark.dist
@pytest.mark.parametrize("num_experts", [4, 64])
@pytest.mark.parametrize("batch_size", [16])
@pytest.mark.parametrize("dim", [64])
@rerun_if_address_is_in_use()
def test_moe_hierarchical_ep(num_experts: int, batch_size: int, dim: int):
    spawn(run_test, 4, num_experts=num_experts, batch_size=batch_size, dim=dim)


if __name__ == "__main__":
    test_moe_hierarchical_ep(num_experts=8, batch_size=8, dim=32)
