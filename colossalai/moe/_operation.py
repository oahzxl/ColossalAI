from typing import Any, Optional, Tuple

import torch
import torch.distributed as dist
from einops import rearrange
from torch import Tensor
from torch.cuda.amp import custom_bwd, custom_fwd
from torch.distributed import ProcessGroup

MOE_KERNEL = None


def load_moe():
    global MOE_KERNEL
    from colossalai.kernel.op_builder import MOEBuilder

    MOE_KERNEL = MOEBuilder().load()


class AllGather(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        inputs: Tensor,
        group: Optional[ProcessGroup] = None,
        overlap: bool = False,
    ) -> Tuple[Tensor, Any]:
        """
        Returns:
            outputs: Tensor
            handle: Optional[Work], if overlap is True
        """
        assert ctx is not None or not overlap

        if ctx is not None:
            ctx.comm_grp = group

        comm_size = dist.get_world_size(group)
        if comm_size == 1:
            return inputs.unsqueeze(0), None

        buffer_shape = (comm_size,) + inputs.shape
        outputs = torch.empty(buffer_shape, dtype=inputs.dtype, device=inputs.device)
        buffer_list = list(torch.chunk(outputs, comm_size, dim=0))
        if not overlap:
            dist.all_gather(buffer_list, inputs, group=group)
            return outputs, None
        else:
            handle = dist.all_gather(buffer_list, inputs, group=group, async_op=True)
            return outputs, handle

    @staticmethod
    def backward(ctx: Any, *grad_outputs) -> Tuple[Tensor, None, None]:
        return (
            ReduceScatter.forward(None, grad_outputs[0], ctx.comm_grp, False)[0],
            None,
            None,
        )


class ReduceScatter(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        inputs: Tensor,
        group: ProcessGroup,
        overlap: bool = False,
    ) -> Tuple[Tensor, Any]:
        """
        Returns:
            outputs: Tensor
            handle: Optional[Work], if overlap is True
        """
        assert ctx is not None or not overlap

        if ctx is not None:
            ctx.comm_grp = group

        comm_size = dist.get_world_size(group)
        if comm_size == 1:
            return inputs.squeeze(0), None

        if not inputs.is_contiguous():
            inputs = inputs.contiguous()

        output_shape = inputs.shape[1:]
        outputs = torch.empty(output_shape, dtype=inputs.dtype, device=inputs.device)
        buffer_list = list(torch.chunk(inputs, comm_size, dim=0))
        if not overlap:
            dist.reduce_scatter(outputs, buffer_list, group=group)
            return outputs, None
        else:
            handle = dist.reduce_scatter(outputs, buffer_list, group=group, async_op=True)
            return outputs, handle

    @staticmethod
    def backward(ctx: Any, *grad_outputs) -> Tuple[Tensor, None, None]:
        # TODO: support async backward
        return (
            AllGather.forward(None, grad_outputs[0], ctx.comm_grp, False)[0],
            None,
            None,
        )


class AllToAll(torch.autograd.Function):
    """Dispatches input tensor [e, c, h] to all experts by all_to_all_single
    operation in torch.distributed.
    """

    @staticmethod
    def forward(
        ctx: Any,
        inputs: Tensor,
        group: ProcessGroup,
        overlap: bool = False,
    ) -> Tuple[Tensor, Any]:
        """
        Returns:
            outputs: Tensor
            handle: Optional[Work], if overlap is True
        """
        assert ctx is not None or not overlap

        if ctx is not None:
            ctx.comm_grp = group
        if not inputs.is_contiguous():
            inputs = inputs.contiguous()
        if dist.get_world_size(group) == 1:
            return inputs, None
        output = torch.empty_like(inputs)
        if not overlap:
            dist.all_to_all_single(output, inputs, group=group)
            return output, None
        else:
            handle = dist.all_to_all_single(output, inputs, group=group, async_op=True)
            return output, handle

    @staticmethod
    def backward(ctx: Any, *grad_outputs) -> Tuple[Tensor, None, None]:
        return (
            AllToAll.forward(None, grad_outputs[0], ctx.comm_grp, False)[0],
            None,
            None,
        )


class HierarchicalAllToAll(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        inputs: Tensor,
        ep_info: Tuple[int, ProcessGroup, ProcessGroup, int, int],
        used_capacity: Tensor,
        enable_dp_balance: bool,
        send: bool,
        capacity: int,
        capacity_count: Tensor,
        num_experts: int,
    ) -> Tensor:
        """
        Args:
            inputs (torch.Tensor): (num_experts, capacity, hidden_size)
            used_capacity (torch.Tensor): (num_experts,)
            ep_info (Tuple[int, ProcessGroup, ProcessGroup, int, int]): (src_rank, intra_ep_group, inter_ep_group, local_ep_size, experts_per_gpu)

        Returns:
            outputs: Tensor
        """
        # TODO: we can reduce comm volume by removing empty capacity
        src_rank, intra_ep_group, inter_ep_group, local_ep_size, experts_per_gpu = ep_info
        if ctx is not None:
            ctx.ep_info = ep_info
            ctx.used_capacity = used_capacity
            ctx.enable_dp_balance = enable_dp_balance
            ctx.send = not send
            ctx.capacity = capacity
            ctx.num_experts = num_experts

        local_gpu_num = dist.get_world_size(intra_ep_group)
        dp_size = local_gpu_num // local_ep_size

        # add used_capacity to the last cap of inputs
        if enable_dp_balance and send:
            inputs[:, -1, 0] = used_capacity

        # master node
        if dist.get_rank() == src_rank:
            # intra-node gather
            inter_ep_size = dist.get_world_size(inter_ep_group)
            intra_gather = [torch.empty_like(inputs) for _ in range(local_gpu_num)]
            dist.gather(inputs, intra_gather, dst=src_rank, group=intra_ep_group)

            # inter-node all-to-all
            if inter_ep_size == 1:
                inter_all2all = intra_gather
            else:
                # layout transform before all2all
                if enable_dp_balance and not send:
                    # current: [(local_ep_size dp_size) expert_per_gpu squeezed_dim h]
                    # target:  [inter_ep_size (local_gpu_num local_expert_num) cap h]
                    intra_gather = dp_balance_recv(
                        dp_size, inter_ep_size, local_ep_size, experts_per_gpu, capacity, capacity_count, intra_gather
                    )
                else:
                    # inputs: (expert_num, cap, h)
                    # intra_gather: (local_gpu_num, expert_num, cap, h)
                    # current: [local_gpu_num expert_num cap h]
                    #        = [local_gpu_num (inter_ep_size local_expert_num) cap h]
                    # target:  [inter_ep_size (local_gpu_num local_expert_num) cap h]
                    intra_gather = rearrange(
                        intra_gather,
                        "local_gpu_num (inter_ep_size local_expert_num) cap h -> inter_ep_size (local_gpu_num local_expert_num) cap h",
                        inter_ep_size=inter_ep_size,
                    ).contiguous()

                # all2all
                inter_all2all = torch.empty_like(intra_gather)
                dist.all_to_all_single(inter_all2all, intra_gather, group=inter_ep_group)

                # layout transform after all2all
                if enable_dp_balance and send:
                    # current: [inter_ep_size (local_gpu_num local_expert_num) cap h]
                    #       =  [inter_ep_size (local_ep_size1 dp_size local_ep_size2 expert_per_gpu) cap h]
                    # target:  [(local_ep_size dp_size) expert_per_gpu squeezed_dim h]
                    inter_all2all, capacity_count, max_seq_len = dp_balance_send(
                        dp_size, local_ep_size, experts_per_gpu, capacity_count, inter_all2all
                    )
                else:
                    # current: [inter_ep_size (local_gpu_num local_expert_num) cap h]
                    #       =  [inter_ep_size (local_ep_size1 dp_size local_ep_size2 expert_per_gpu) cap h]
                    # target:  [(local_ep_size2 dp_size) (inter_ep_size local_ep_size1 expert_per_gpu) cap h]
                    #       =  [local_gpu_num expert_num cap h]
                    inter_all2all = rearrange(
                        inter_all2all,
                        "inter_ep_size (local_ep_size1 dp_size local_ep_size2 expert_per_gpu) cap h -> (local_ep_size2 dp_size) (inter_ep_size local_ep_size1 expert_per_gpu) cap h",
                        expert_per_gpu=experts_per_gpu,
                        local_ep_size1=local_ep_size,
                        local_ep_size2=local_ep_size,
                    )

            # intra-node scatter
            intra_scatter = list(inter_all2all.chunk(local_gpu_num, dim=0))
            intra_scatter = [i.squeeze(0).contiguous() for i in intra_scatter]
            if enable_dp_balance and send:
                # the shape is dynamic for send
                new_seq_len = max_seq_len // dp_size
                dist.broadcast(
                    torch.tensor(new_seq_len, device=inputs.device, dtype=torch.long),
                    src=src_rank,
                    group=intra_ep_group,
                )
                outputs = torch.empty(
                    (experts_per_gpu, new_seq_len, inputs.shape[-1]), device=inputs.device, dtype=inputs.dtype
                )
            else:
                outputs = torch.empty(
                    (num_experts, capacity, inputs.shape[-1]), device=inputs.device, dtype=inputs.dtype
                )
            dist.scatter(outputs, intra_scatter, src=src_rank, group=intra_ep_group)

        # slave node
        else:
            # send inputs
            dist.gather(inputs, dst=src_rank, group=intra_ep_group)
            # get sequence length
            if enable_dp_balance and send:
                # the shape is dynamic for send
                shape = torch.tensor(0, device=inputs.device, dtype=torch.long)
                dist.broadcast(shape, src=src_rank, group=intra_ep_group)
                outputs = torch.empty(
                    (experts_per_gpu, int(shape), inputs.shape[-1]), device=inputs.device, dtype=inputs.dtype
                )
            else:
                outputs = torch.empty(
                    (num_experts, capacity, inputs.shape[-1]), device=inputs.device, dtype=inputs.dtype
                )
            # receive outputs
            dist.scatter(outputs, src=src_rank, group=intra_ep_group)

        if ctx is not None:
            ctx.capacity_count = capacity_count
            # add used_capacity for reuse
            if enable_dp_balance and send:
                outputs = (outputs, capacity_count)

        return outputs

    @staticmethod
    def backward(ctx: Any, *grad_outputs) -> Tuple[Tensor, None, None]:
        return (
            HierarchicalAllToAll.forward(
                None,
                grad_outputs[0],
                ctx.ep_info,
                ctx.used_capacity,
                ctx.enable_dp_balance,
                ctx.send,
                ctx.capacity,
                ctx.capacity_count,
                ctx.num_experts,
            ),
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


def dp_balance_recv(dp_size, inter_ep_size, local_ep_size, experts_per_gpu, capacity, capacity_count, intra_gather):
    intra_gather = rearrange(
        intra_gather,
        "(local_ep_size dp_size) expert_per_gpu other h -> (local_ep_size expert_per_gpu) (dp_size other) h",
        dp_size=dp_size,
    )
    # intra_gather: (local_ep_size dp_size) expert_per_gpu other h
    # target: (local_ep_size2 expert_per_gpu) (dp_size inter_ep_size local_ep_size1) cap h
    new_out = torch.zeros(
        (
            local_ep_size * experts_per_gpu,
            dp_size * inter_ep_size * local_ep_size,
            capacity,
            intra_gather.shape[-1],
        ),
        device=intra_gather.device,
        dtype=intra_gather.dtype,
    )
    # capacity_count: (local_ep_size2 expert_per_gpu) (dp_size inter_ep_size local_ep_size1)
    for l in range(new_out.shape[0]):
        cap_idx = 0
        seq_pointer = 0
        while cap_idx < capacity_count.shape[1]:
            seq_len = int(capacity_count[l, cap_idx])
            new_out[l, cap_idx, :seq_len] = intra_gather[l, seq_pointer : seq_pointer + seq_len]
            seq_pointer += seq_len
            cap_idx += 1
    intra_gather = rearrange(
        new_out,
        "(local_ep_size1 expert_per_gpu) (dp_size inter_ep_size local_ep_size2) cap h -> inter_ep_size (local_ep_size1 dp_size local_ep_size2 expert_per_gpu) cap h",
        dp_size=dp_size,
        local_ep_size1=local_ep_size,
        local_ep_size2=local_ep_size,
    ).contiguous()
    return intra_gather


def dp_balance_send(dp_size, local_ep_size, experts_per_gpu, capacity_count, inter_all2all):
    # layout transform after all2all
    # current: [inter_ep_size (local_gpu_num local_expert_num) cap h]
    #       =  [inter_ep_size (local_ep_size1 dp_size local_ep_size2 expert_per_gpu) cap h]
    # target:  [(local_ep_size2 dp_size) (inter_ep_size local_ep_size1 expert_per_gpu) cap h]
    #       =  [local_gpu_num expert_num cap h]
    inter_all2all = rearrange(
        inter_all2all,
        "inter_ep_size (local_ep_size1 dp_size local_ep_size2 expert_per_gpu) cap h -> (local_ep_size2 expert_per_gpu) (dp_size inter_ep_size local_ep_size1) cap h",
        expert_per_gpu=experts_per_gpu,
        local_ep_size1=local_ep_size,
        local_ep_size2=local_ep_size,
    )
    # capacity_count: (local_ep_size2 expert_per_gpu) (dp_size inter_ep_size local_ep_size1)
    capacity_count = inter_all2all[:, :, -1, 0]
    # capacity_sum: local_ep_size2 expert_per_gpu
    capacity_sum = capacity_count.sum(dim=1)
    capacity_mean = capacity_sum / dp_size
    new_tensor = []
    for l in range(local_ep_size * experts_per_gpu):
        ep_tensor_list = []
        ep_cap_pos = 0
        ep_cap_count = 0
        for _ in range(dp_size):
            dp_tensor_list = []

            while True:
                # end of capacity
                if ep_cap_pos == capacity_count.shape[1]:
                    break
                if ep_cap_count >= 0.95 * capacity_mean[l]:
                    ep_cap_count = 0
                    break
                # add tokens
                seq_len = int(capacity_count[l, ep_cap_pos])
                dp_tensor_list.append(inter_all2all[l, ep_cap_pos, :seq_len])
                # add current capacity
                ep_cap_count += seq_len
                ep_cap_pos += 1

            # add dp_tensor to ep_tensor
            ep_tensor_list.append(torch.cat(dp_tensor_list, dim=0))

        new_tensor.append(torch.cat(ep_tensor_list, dim=0))

    # calculate max_len
    max_len = max([i.shape[0] for i in new_tensor])
    if max_len % dp_size != 0:
        max_len = max_len + dp_size - max_len % dp_size

    for i in range(len(new_tensor)):
        if new_tensor[i].shape[0] < max_len:
            new_tensor[i] = torch.nn.functional.pad(new_tensor[i], (0, 0, 0, max_len - new_tensor[i].shape[0]))
    new_tensor = torch.cat(new_tensor, dim=0)
    new_tensor = rearrange(
        new_tensor,
        "(local_ep_size expert_per_gpu dp_size other) h -> (local_ep_size dp_size) expert_per_gpu other h",
        expert_per_gpu=experts_per_gpu,
        local_ep_size=local_ep_size,
        dp_size=dp_size,
    )
    return new_tensor, capacity_count, max_len


class MoeDispatch(torch.autograd.Function):
    @staticmethod
    @custom_fwd
    def forward(ctx, tokens, mask, dest_idx, ec):
        s = tokens.size(0)
        h = tokens.size(1)
        dtype = tokens.dtype

        if MOE_KERNEL is None:
            load_moe()
        if tokens.dtype != torch.float32:
            tokens = tokens.to(torch.float32)
        expert_input = MOE_KERNEL.dispatch_forward(s, ec, h, tokens, mask, dest_idx)
        if expert_input.dtype != dtype:
            expert_input = expert_input.to(dtype)
        ctx.save_for_backward(mask, dest_idx)
        ctx.s = s
        ctx.h = h
        ctx.ec = ec
        ctx.dtype = dtype

        return expert_input

    @staticmethod
    @custom_bwd
    def backward(ctx, output_grad):
        mask, dest_idx = ctx.saved_tensors
        if output_grad.dtype != torch.float32:
            output_grad = output_grad.to(torch.float32)
        d_tokens = MOE_KERNEL.dispatch_backward(ctx.s, ctx.ec, ctx.h, output_grad, mask, dest_idx)
        if d_tokens.dtype != ctx.dtype:
            d_tokens = d_tokens.to(ctx.dtype)
        return d_tokens, None, None, None


class MoeCombine(torch.autograd.Function):
    @staticmethod
    @custom_fwd
    def forward(ctx, expert_tokens, logits, mask, dest_idx, ec):
        assert logits.dtype == torch.float32

        s = logits.size(0)
        e = logits.size(1)
        c = ec // e
        h = expert_tokens.size(-1)
        dtype = expert_tokens.dtype

        if expert_tokens.dtype != torch.float32:
            expert_tokens = expert_tokens.to(torch.float32)
        if MOE_KERNEL is None:
            load_moe()
        output = MOE_KERNEL.combine_forward(s, e, c, h, expert_tokens, logits, mask, dest_idx)
        if output.dtype != dtype:
            output = output.to(dtype)

        ctx.save_for_backward(expert_tokens, logits, mask, dest_idx)
        ctx.s = s
        ctx.e = e
        ctx.c = c
        ctx.h = h
        ctx.dtype = dtype

        return output

    @staticmethod
    @custom_bwd
    def backward(ctx, tokens_grad):
        expert_tokens, logits, mask, dest_idx = ctx.saved_tensors
        if tokens_grad.dtype != torch.float32:
            tokens_grad = tokens_grad.to(torch.float32)

        d_expert, d_logits = MOE_KERNEL.combine_backward(
            ctx.s, ctx.e, ctx.c, ctx.h, tokens_grad, expert_tokens, logits, mask, dest_idx
        )
        if d_expert.dtype != ctx.dtype:
            d_expert = d_expert.to(ctx.dtype)

        return d_expert, d_logits, None, None, None


def moe_cumsum(inputs: Tensor, use_kernel: bool = False):
    dim0 = inputs.size(0)
    flag = (dim0 <= 1024) or (dim0 <= 2048 and dim0 % 2 == 0) or (dim0 % 4 == 0)
    if flag and use_kernel:
        if MOE_KERNEL is None:
            load_moe()
        return MOE_KERNEL.cumsum_sub_one(inputs)
    else:
        return torch.cumsum(inputs, dim=0) - 1


class MoeInGradScaler(torch.autograd.Function):
    """
    Scale the gradient back by the number of experts
    because the batch size increases in the moe stage
    """

    @staticmethod
    def forward(ctx: Any, inputs: Tensor, ep_size: int) -> Tensor:
        if ctx is not None:
            ctx.ep_size = ep_size
        return inputs

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Tensor) -> Tuple[Tensor, None]:
        assert len(grad_outputs) == 1
        grad = grad_outputs[0]
        if ctx.ep_size != 1:
            grad = grad * ctx.ep_size
        return grad, None


class MoeOutGradScaler(torch.autograd.Function):
    """
    Scale the gradient by the number of experts
    because the batch size increases in the moe stage
    """

    @staticmethod
    def forward(ctx: Any, inputs: Tensor, ep_size: int) -> Tensor:
        ctx.ep_size = ep_size
        return inputs

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Tensor) -> Tuple[Tensor, None]:
        assert len(grad_outputs) == 1
        grad = grad_outputs[0]
        if ctx.ep_size != 1:
            grad = grad / ctx.ep_size
        return grad, None
