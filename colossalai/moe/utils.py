import contextlib
import os
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from colossalai.moe.manager import MOE_MANAGER
from colossalai.tensor.moe_tensor.api import get_dp_group, get_dp_group_ranks, get_ep_size, is_moe_tensor
from colossalai.utils import get_current_device


class ForceFP32Parameter(torch.nn.Parameter):
    def half(self, memory_format=None):
        return self.data.clone()


class NormalNoiseGenerator:
    """Generates a random noisy mask for logits tensor.

    All noise is generated from a normal distribution :math:`(0, 1 / E^2)`, where
    `E = the number of experts`.

    Args:
        num_experts (int): The number of experts.
    """

    def __init__(self, num_experts: int):
        self.normal = torch.distributions.normal.Normal(
            loc=torch.tensor(0.0, device=get_current_device()),
            scale=torch.tensor(1.0 / num_experts**2, device=get_current_device()),
        ).rsample

    def __call__(self, inputs: torch.Tensor):
        noisy = self.normal(inputs.shape)
        return inputs + noisy


class UniformNoiseGenerator:
    """Generates a random noisy mask for logits tensor.
    copied from mesh tensorflow:
    Multiply values by a random number between :math:`1-epsilon` and :math:`1+epsilon`.
    Makes models more resilient to rounding errors introduced by bfloat16.
    This seems particularly important for logits.

    Args:
        eps (float, optional): Epsilon in generator, defaults 1e-2.
    """

    def __init__(self, eps: float = 1e-2):
        self.uniform = torch.distributions.uniform.Uniform(
            low=torch.tensor(1.0 - eps, device=get_current_device()),
            high=torch.tensor(1.0 + eps, device=get_current_device()),
        ).rsample

    def __call__(self, inputs: torch.Tensor):
        noisy = self.uniform(inputs.shape)
        return inputs * noisy


def autocast_softmax(logit: torch.Tensor, dim: int):
    return F.softmax(logit, dim=dim, detype=torch.float32)


def get_noise_generator(noise_type: str, num_experts: int) -> Callable:
    if noise_type is None:
        return None
    elif noise_type == "Jitter":
        noisy_func = UniformNoiseGenerator()
    elif noise_type == "Gaussian":
        noisy_func = NormalNoiseGenerator(num_experts)
    else:
        raise NotImplementedError("Unsupported input noisy policy")
    return noisy_func


def get_activation(act: str) -> Callable:
    if act is None or act == "relu":
        return torch.nn.ReLU()
    elif act == "gelu":
        return torch.nn.GELU()
    elif act == "swiglu":
        return SwiGLU
    elif act == "silu":
        return torch.nn.SiLU()
    else:
        raise NotImplementedError("Unsupported activation function")


def SwiGLU(x):
    """Gated linear unit activation function.
    Args:
        x : input array
        axis: the axis along which the split should be computed (default: -1)
    """
    size = x.shape[-1]
    assert size % 2 == 0, "axis size must be divisible by 2"
    x1, x2 = torch.split(x, size // 2, -1)
    return x1 * (x2 * torch.sigmoid(x2))


@contextlib.contextmanager
def skip_init():
    """
    skip param random init
    """

    def _skip_init(*args, **kwargs):
        pass

    init_func = {
        "constant_": torch.nn.init.constant_,
        "uniform_": torch.nn.init.uniform_,
        "normal_": torch.nn.init.normal_,
        "kaiming_uniform_": torch.nn.init.kaiming_uniform_,
        "kaiming_normal_": torch.nn.init.kaiming_normal_,
        "xavier_normal_": torch.nn.init.xavier_normal_,
        "xavier_uniform_": torch.nn.init.xavier_uniform_,
        "trunc_normal_": torch.nn.init.trunc_normal_,
    }

    for method_name, original_init in init_func.items():
        setattr(torch.nn.init, method_name, _skip_init)

    yield

    for method_name, original_init in init_func.items():
        setattr(torch.nn.init, method_name, original_init)

    return


def get_moe_epsize_param_dict(model: nn.Module) -> Dict[int, List[nn.Parameter]]:
    """Returns a parameter dictionary, the key of which is the expert parallel
    size of every parameter. Since the parameters in data parallelism is replicated
    in each GPU, we set their ep_size to 1.

    Args:
        model (:class:`torch.nn.Module`): A pyTorch `nn.Module` from which we get dict.
    """
    epsize_param_dict = dict()
    for param in model.parameters():
        if not is_moe_tensor(param):
            ep_size = 1  # set ep_size to 1 for dp parameters
        else:
            ep_size = get_ep_size(param)
        if ep_size not in epsize_param_dict:
            epsize_param_dict[ep_size] = []
        epsize_param_dict[ep_size].append(param)

    return epsize_param_dict


def sync_moe_model_param(model: nn.Module):
    """Make sure model parameters are consistent in MoE parallel context.

    Args:
        model (:class:`torch.nn.Module`): A pyTorch model on whose parameters you check the consistency.
    """
    param_dict = get_moe_epsize_param_dict(model)

    # synchronize the parameters whose dp_group is the whole world
    if 1 in param_dict:
        for param in param_dict[1]:
            dist.broadcast(param, src=0)

    for ep_size in param_dict:
        # When ep_size = world_size, communication is not needed
        if ep_size != 1 and ep_size != MOE_MANAGER.world_size:
            for param in param_dict[ep_size]:
                src_rank = get_dp_group_ranks(param)[0]
                dist.broadcast(param, src=src_rank, group=get_dp_group(param))


def set_moe_args(config: Any, args: dict):
    for k, v in args.items():
        setattr(config, k, v)


def create_ep_hierarchical_group(
    ep_group_ranks: List[int],
    experts_per_gpu: int,
    nproc_per_node: Optional[int] = None,
) -> Tuple[int, dist.ProcessGroup, Optional[dist.ProcessGroup]]:
    assert dist.is_initialized(), "Please initialize torch.distributed first."
    rank = dist.get_rank()
    if nproc_per_node is None:
        nproc_per_node = os.environ.get("LOCAL_WORLD_SIZE")
        assert nproc_per_node is not None, "Please use torchrun to launch the job, or specify nproc_per_node manually."
        nproc_per_node = int(nproc_per_node)
    else:
        assert dist.get_world_size() % nproc_per_node == 0, "nproc_per_node should be a divisor of world_size."
    num_node = dist.get_world_size() // nproc_per_node

    # master node will gather and scatter all data in a node, and all2all with other master nodes
    master_node_ranks = [i * nproc_per_node for i in range(num_node)]
    master_node_group = dist.new_group(master_node_ranks)

    # the master node smaller than current rank is the intra_src_rank
    intra_src_rank = [i > rank for i in master_node_ranks]
    if True not in intra_src_rank:
        intra_src_rank = master_node_ranks[-1]
        if len(master_node_ranks) == 1:
            local_ep_size = nproc_per_node // len(ep_group_ranks)
        else:
            master_node_rank_gap = master_node_ranks[-1] - master_node_ranks[-2]
            local_ep_size = sum(
                [master_node_ranks[-1] <= i < (master_node_ranks[-1] + master_node_rank_gap) for i in ep_group_ranks]
            )
    else:
        intra_src_rank_idx = intra_src_rank.index(True)
        intra_src_rank = master_node_ranks[intra_src_rank_idx - 1]
        local_ep_size = sum(
            [
                master_node_ranks[intra_src_rank_idx - 1] <= i < master_node_ranks[intra_src_rank_idx]
                for i in ep_group_ranks
            ]
        )

    # ep intra ranks are the ranks in the same node
    for i in range(0, dist.get_world_size(), nproc_per_node):
        cur_intra_node_ranks = [i + j for j in range(nproc_per_node)]
        # need to create group for each intra_src_rank
        cur_intra_node_group = dist.new_group(cur_intra_node_ranks)
        if i == intra_src_rank:
            intra_node_group = cur_intra_node_group

    # print(
    #     f"rank {rank} intra_src_rank {intra_src_rank} intra_node_ranks {intra_node_ranks} master_node_ranks {master_node_ranks} local_ep_size {local_ep_size} experts_per_gpu {experts_per_gpu}"
    # )
    return intra_src_rank, intra_node_group, master_node_group, local_ep_size, experts_per_gpu


def create_tp_balance_group(
    ep_group_ranks: List[int],
    nproc_per_node: Optional[int] = None,
) -> Tuple[int, dist.ProcessGroup, Optional[dist.ProcessGroup]]:
    """
    create tensor parallel group for expert balance

    e.g. for 4 gpus [1, 2, 3, 4]. [1, 2] are first ep group, [3, 4] are second ep group.
        and dp group is within each ep group. Then the tp group is [1, 3] and [2, 4] or
        it can be [1, 2, 3, 4].
    """
    assert dist.is_initialized(), "Please initialize torch.distributed first."
    rank = dist.get_rank()
    if nproc_per_node is None:
        nproc_per_node = os.environ.get("LOCAL_WORLD_SIZE")
        assert nproc_per_node is not None, "Please use torchrun to launch the job, or specify nproc_per_node manually."
        nproc_per_node = int(nproc_per_node)
    else:
        assert dist.get_world_size() % nproc_per_node == 0, "nproc_per_node should be a divisor of world_size."
    num_node = dist.get_world_size() // nproc_per_node

    tp_size_cross_ep = 2
    tp_size_cross_dp = 1

    # master node will gather and scatter all data in a node, and all2all with other master nodes
    master_node_ranks = [i * nproc_per_node for i in range(num_node)]

    # the master node smaller than current rank is the intra_src_rank
    intra_src_rank = [i > rank for i in master_node_ranks]
    if True not in intra_src_rank:
        intra_src_rank = master_node_ranks[-1]
        if len(master_node_ranks) == 1:
            local_ep_size = nproc_per_node // len(ep_group_ranks)
        else:
            master_node_rank_gap = master_node_ranks[-1] - master_node_ranks[-2]
            local_ep_size = sum(
                [master_node_ranks[-1] <= i < (master_node_ranks[-1] + master_node_rank_gap) for i in ep_group_ranks]
            )
    else:
        intra_src_rank_idx = intra_src_rank.index(True)
        intra_src_rank = master_node_ranks[intra_src_rank_idx - 1]
        local_ep_size = sum(
            [
                master_node_ranks[intra_src_rank_idx - 1] <= i < master_node_ranks[intra_src_rank_idx]
                for i in ep_group_ranks
            ]
        )
    local_dp_size = nproc_per_node // local_ep_size

    # ep intra ranks are the ranks in the same node
    assert (
        local_ep_size % tp_size_cross_ep == 0
    ), f"local_ep_size should be divisible by tp_size_cross_ep, {local_ep_size} % {tp_size_cross_ep} = {local_ep_size % tp_size_cross_ep}"
    assert (
        local_dp_size % tp_size_cross_dp == 0
    ), f"local_dp_size should be divisible by tp_size_cross_dp, {local_dp_size} % {tp_size_cross_dp} = {local_dp_size % tp_size_cross_dp}"
    dp_iter = local_dp_size // tp_size_cross_dp
    ep_iter = local_ep_size // tp_size_cross_ep
    tp_balance_group = None
    for i in range(0, dist.get_world_size(), nproc_per_node):
        cur_intra_node_ranks = [i + j for j in range(nproc_per_node)]
        cur_ep_group_ranks = [
            cur_intra_node_ranks[j : j + local_dp_size]
            for j in range(0, len(cur_intra_node_ranks), nproc_per_node // local_ep_size)
        ]
        for m in range(ep_iter):
            cur_ep_rank = cur_ep_group_ranks[m * tp_size_cross_ep : (m + 1) * tp_size_cross_ep]
            for n in range(dp_iter):
                cur_dp_rank_in_ep = [
                    cur_ep_rank[k][n * tp_size_cross_dp : (n + 1) * tp_size_cross_dp] for k in range(tp_size_cross_ep)
                ]
                cur_tp_rank = sum(cur_dp_rank_in_ep, [])
                cur_tp_group = dist.new_group(cur_tp_rank)
                if rank in cur_tp_rank:
                    tp_balance_group = cur_tp_group
    assert tp_balance_group is not None, "tp_balance_group should not be None"
    # print(f"rank {rank}: tp_balance_rank {tp_balance_rank}")
    return tp_balance_group
