import os

import pytest
import torch
import torch.distributed as dist

import colossalai
from colossalai.moe import SparseMLP
from colossalai.moe.manager import MOE_MANAGER
from colossalai.moe.utils import sync_moe_model_param
from colossalai.tensor.moe_tensor.api import get_ep_group, get_ep_size
from colossalai.testing import assert_equal_in_group, rerun_if_address_is_in_use, spawn
from colossalai.utils import get_current_device
from tests.test_moe.moe_utils import MoeGradientHandler


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
    local_model = SparseMLP(
        num_experts=num_experts, hidden_size=dim, intermediate_size=dim * 2, router_capacity_factor_train=batch_size
    )
    MOE_MANAGER.__init__()
    MOE_MANAGER.setup(parallel="EP", use_ep_inside=False)
    # fake to be 2 nodes
    os.environ["LOCAL_WORLD_SIZE"] = str(world_size // 2)
    ep_model = SparseMLP(
        num_experts=num_experts,
        hidden_size=dim,
        intermediate_size=dim * 2,
        enable_hierarchical_comm=True,
        router_capacity_factor_train=batch_size,
    )
    ep_model = ep_model.to(get_current_device()).train()
    local_model = local_model.to(get_current_device()).train()

    # sync ep param
    sync_moe_model_param(ep_model)
    dist_dict = MOE_MANAGER.parallel_info_dict
    assert_equal_in_group(ep_model.experts.wi.data, dist_dict[world_size].dp_group)
    assert_equal_in_group(ep_model.experts.wo.data, dist_dict[world_size].dp_group)
    ep_grad_handler = MoeGradientHandler(ep_model)
    # sync local param
    sync_local_from_ep(local_model, ep_model)

    torch.manual_seed(0)
    rank = dist.get_rank()
    input_data = torch.randn(batch_size, dim, device=get_current_device())
    micro_batch_size = batch_size // world_size
    index = rank * micro_batch_size
    # NOTE: ep & tp takes in sharded data for each process
    shard_data = input_data.detach()[index : index + micro_batch_size]

    out_local = local_model(input_data)
    MOE_MANAGER.reset_loss()
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
@pytest.mark.parametrize(
    "config",
    [
        {"world_size": 4, "num_experts": 4, "batch_size": 8, "dim": 4},
        {"world_size": 4, "num_experts": 8, "batch_size": 32, "dim": 4},
    ],
)
@rerun_if_address_is_in_use()
def test_moe_hierarchical_ep(config: dict):
    spawn(
        run_test,
        config["world_size"],
        num_experts=config["num_experts"],
        batch_size=config["batch_size"],
        dim=config["dim"],
    )


if __name__ == "__main__":
    test_moe_hierarchical_ep({"world_size": 4, "num_experts": 8, "batch_size": 8, "dim": 4})
