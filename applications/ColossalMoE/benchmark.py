import argparse
import os
from time import time
from typing import Optional

import torch
import torch.distributed as dist
import torch.nn as nn
from colossal_moe.models.mixtral_checkpoint import MixtralMoECheckpointIO
from colossal_moe.models.mixtral_layer import replace_moe_layer
from colossal_moe.models.mixtral_policy import MixtralForCausalLMPolicy
from colossal_moe.utils import move_to_cuda
from torch import Tensor
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers.models.mixtral import MixtralConfig, MixtralForCausalLM

import colossalai
from colossalai.booster import Booster
from colossalai.booster.plugin.moe_hybrid_parallel_plugin import MoeHybridParallelPlugin
from colossalai.cluster import DistCoordinator
from colossalai.logging import DistributedLogger
from colossalai.moe import MOE_MANAGER, apply_load_balance
from colossalai.nn.optimizer import HybridAdam
from colossalai.utils import get_current_device


@torch.no_grad()
def get_global_loss(loss, booster):
    global_loss = loss.clone().detach()
    dist.all_reduce(tensor=global_loss, op=dist.ReduceOp.SUM, group=booster.plugin.dp_group)
    global_loss.div_(booster.plugin.dp_size)
    return global_loss


class RandomDataset(Dataset):
    def __init__(self, num_samples: int = 1000, max_length: int = 2048, vocab_size: int = 100, tokenizer=None):
        self.num_samples = num_samples
        self.max_length = max_length
        self.input_ids = torch.randint(0, vocab_size, (num_samples, max_length), device=get_current_device())
        self.attention_mask = torch.ones_like(self.input_ids)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "labels": self.input_ids[idx],
        }


def parse_args():
    # basic settings
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument("--load_checkpoint", type=str, default=None, help="Load checkpoint")
    parser.add_argument(
        "--plugin",
        type=str,
        help="Parallel methods.",
    )
    parser.add_argument("--num_epoch", type=int, default=1, help="Number of epochs.")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size (per dp group) for the training dataloader.",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="bf16",
        choices=["fp32", "bf16", "fp16"],
        help="The mixed precision training.",
    )
    parser.add_argument("--max_length", type=int, default=2048, help="Max sequence length.")
    parser.add_argument("--seed", type=int, default=42, help="A seed for reproducible training.")

    # bench
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--active", type=int, default=20)
    parser.add_argument("--half_local_world_size", action="store_true", help="Half local world size.")

    # zero stage for all plugins
    parser.add_argument("--zero_stage", type=int, default=2, help="zero stage.")

    # balance
    parser.add_argument("--enable_dp_balance", action="store_true", help="Enable dp balance.")
    parser.add_argument("--enable_tp_balance", action="store_true", help="Enable tp balance.")

    # hybrid plugin
    parser.add_argument("--extra_dp_size", type=int, default=1)
    parser.add_argument("--pp_size", type=int, default=2, help="pp size for hybrid plugin")
    parser.add_argument("--dp_size", type=int, default=1, help="dp size for hybrid plugin")
    parser.add_argument("--ep_size", type=int, default=2, help="ep size for hybrid plugin")
    parser.add_argument("--microbatch_size", type=int, default=1, help="Microbatch size in pipeline for hybrid plugin")

    # kernel
    parser.add_argument(
        "--use_kernel",
        action="store_true",
        help="Use kernel optim. Need to install flash attention and triton to enable all kernel optimizations. Skip if not installed.",
    )
    parser.add_argument(
        "--use_layernorm_kernel",
        action="store_true",
        help="Use layernorm kernel. Need to install apex. Raise error if not installed.",
    )

    # load balance
    parser.add_argument(
        "--load_balance", action="store_true", help="Expert load balance. Defaults to False. Recommend to enable."
    )
    parser.add_argument("--load_balance_interval", type=int, default=1000, help="Expert load balance interval.")
    # communicate overlap
    parser.add_argument(
        "--comm_overlap",
        action="store_true",
        help="Use communication overlap for MoE. Recommended to enable for muiti-node training.",
    )
    # hierarchical all-to-all
    parser.add_argument(
        "--hierarchical_alltoall",
        action="store_true",
        help="Use hierarchical all-to-all for MoE. Recommended to enable for muiti-node training.",
    )

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    # Launch ColossalAI
    colossalai.launch_from_torch(config={}, seed=args.seed)
    coordinator = DistCoordinator()

    # Set plugin
    booster_kwargs = {}
    hybrid_dict = {
        "tp_size": 1,
        "custom_policy": MixtralForCausalLMPolicy(),
        "enable_fused_normalization": args.use_layernorm_kernel,
        "enable_jit_fused": args.use_kernel,
        "precision": args.precision,
        "zero_stage": args.zero_stage,
        "checkpoint_io": MixtralMoECheckpointIO,
    }
    mgr_dict = {}
    if args.plugin == "hybrid":
        dp_size = dist.get_world_size() // args.pp_size
        plugin = MoeHybridParallelPlugin(
            pp_size=args.pp_size,
            microbatch_size=args.microbatch_size,
            **hybrid_dict,
        )
        MOE_MANAGER.setup(
            parallel="EP",
            mode="fixed",
            fixed_dp_size=args.dp_size,
            fixed_ep_size=args.ep_size,
            fixed_pp_size=args.pp_size,
            **mgr_dict,
        )
    elif args.plugin == "ep":
        dp_size = dist.get_world_size()
        plugin = MoeHybridParallelPlugin(
            pp_size=1,
            **hybrid_dict,
        )
        MOE_MANAGER.setup(
            parallel="EP",
            max_ep_size=dp_size,
            **mgr_dict,
        )
    elif args.plugin == "ep_zero":
        dp_size = dist.get_world_size()
        use_ep_inside = False
        plugin = MoeHybridParallelPlugin(
            pp_size=1,
            extra_dp_size=args.extra_dp_size,
            use_ep_inside=use_ep_inside,
            **hybrid_dict,
        )
        MOE_MANAGER.setup(
            parallel="EP",
            max_ep_size=dp_size // args.extra_dp_size,
            use_ep_inside=use_ep_inside,
            **mgr_dict,
        )
    else:
        raise ValueError(f"Invalid plugin {args.plugin}")
    coordinator.print_on_master(f"Set plugin as {plugin.__class__.__name__}")

    if args.model_name == "mixtral":
        config = MixtralConfig.from_pretrained("mistralai/Mixtral-8x7B-v0.1")
    elif args.model_name == "8x7b":
        config = MixtralConfig(
            hidden_size=4096,
            intermediate_size=14336,
            num_hidden_layers=32,
            num_attention_heads=32,
            num_key_value_heads=8,
        )
    elif args.model_name == "8x1.3b":
        config = MixtralConfig(
            hidden_size=2048,
            intermediate_size=7168,
            num_hidden_layers=24,
            num_attention_heads=32,
            num_key_value_heads=8,
        )
    elif args.model_name == "test":
        # test
        config = MixtralConfig(
            hidden_size=256,
            intermediate_size=512,
            num_hidden_layers=2,
            num_attention_heads=16,
            num_key_value_heads=4,
        )
    else:
        raise ValueError(f"Invalid model name {args.model_name}")

    config.use_cache = False
    config.num_local_experts = 1
    model = MixtralForCausalLM(config)
    model.num_experts = 8
    model = model.to(torch.bfloat16) if args.precision == "bf16" else model.to(torch.float16)
    model = model.to(get_current_device())
    replace_moe_layer(
        model,
        enable_kernel=args.use_kernel,
        enable_dp_balance=args.enable_dp_balance,
        enable_tp_balance=args.enable_tp_balance,
    )
    coordinator.print_on_master(f"Finish init model with config:\n{config}")

    if args.half_local_world_size:
        os.environ["LOCAL_WORLD_SIZE"] = str(dist.get_world_size() // 2)

    # Enable gradient checkpointing
    model.gradient_checkpointing_enable()

    # Prepare tokenizer and dataloader
    dataset = RandomDataset(
        num_samples=args.batch_size * (args.warmup + args.active + 1) * dp_size,
        max_length=args.max_length,
        vocab_size=32000,
    )
    dataloader = plugin.prepare_dataloader(dataset, batch_size=args.batch_size)

    # Set optimizer
    optimizer = HybridAdam(model_params=model.parameters())

    # Set performance evaluator
    model_numel = get_model_numel(model)
    performance_evaluator = PerformanceEvaluator(
        model_numel,
        enable_grad_checkpoint=True,
        ignore_steps=args.warmup,
        dp_world_size=dp_size,
    )

    # Set booster
    booster = Booster(plugin=plugin, **booster_kwargs)
    model, optimizer, _, dataloader, _ = booster.boost(
        model=model,
        optimizer=optimizer,
        dataloader=dataloader,
    )
    use_pipeline = isinstance(booster.plugin, MoeHybridParallelPlugin) and booster.plugin.pp_size > 1
    is_pp_last_stage = use_pipeline and booster.plugin.stage_manager.is_last_stage()
    coordinator.print_on_master(f"Finish init booster")

    # Start finetuning
    coordinator.print_on_master(f"Start finetuning")
    model.train()
    train_dataloader_iter = iter(dataloader)
    total_len = len(train_dataloader_iter) - 1
    exmaple_data = next(train_dataloader_iter)
    with tqdm(
        range(total_len),
        disable=not coordinator.is_master() if use_pipeline == False else not is_pp_last_stage,
    ) as pbar:
        for step in pbar:
            performance_evaluator.on_step_start(step)
            if use_pipeline:
                # Forward pass
                outputs = booster.execute_pipeline(
                    train_dataloader_iter,
                    model,
                    lambda x, y: x.loss,
                    optimizer,
                    return_loss=True,
                    return_outputs=True,
                )
                # Backward and optimize
                if is_pp_last_stage:
                    loss = outputs["loss"]
                    global_loss = get_global_loss(loss, booster)
                    if coordinator._local_rank == "0":
                        pbar.set_postfix({"Loss": global_loss.item()})
            else:
                # Forward pass
                data = next(train_dataloader_iter)
                data = move_to_cuda(data, torch.cuda.current_device())
                outputs = model(**data)
                loss = outputs["loss"]
                # Backward
                booster.backward(loss, optimizer)
                pbar.set_postfix({"loss": loss.item()})

            optimizer.step()
            optimizer.zero_grad()
            performance_evaluator.on_step_end(exmaple_data["input_ids"])

            # Apply load balance
            if args.load_balance and args.load_balance_interval > 0 and (step + 1) % args.load_balance_interval == 0:
                coordinator.print_on_master(f"Apply load balance")
                apply_load_balance(model, optimizer)
        performance_evaluator.on_fit_end()

    # Finish training
    coordinator.print_on_master(f"Finish training")


def print_model_numel(logger: DistributedLogger, model: nn.Module) -> None:
    B = 1024**3
    M = 1024**2
    K = 1024
    outputs = "Model param count: "
    model_param = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if model_param >= B:
        outputs += f"{model_param / B:.2f} B\n"
    elif model_param >= M:
        outputs += f"{model_param / M:.2f} M\n"
    elif model_param >= K:
        outputs += f"{model_param / K:.2f} K\n"
    else:
        outputs += f"{model_param}\n"
    logger.info(outputs, ranks=[0])


def get_model_numel(model: nn.Module) -> None:
    model_param = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return model_param


def divide(x: float, y: float) -> float:
    if y == 0:
        return float("inf")
    elif y == float("inf"):
        return float("nan")
    return x / y


@torch.no_grad()
def all_reduce_mean(x: float, world_size: int) -> float:
    if world_size == 1:
        return x
    tensor = torch.tensor([x], device=torch.cuda.current_device())
    dist.all_reduce(tensor)
    tensor = tensor / world_size
    return tensor.item()


class Timer:
    def __init__(self) -> None:
        self.start_time: Optional[float] = None
        self.duration: float = 0.0

    def start(self) -> None:
        self.start_time = time()

    def end(self) -> None:
        assert self.start_time is not None
        self.duration += time() - self.start_time
        self.start_time = None

    def reset(self) -> None:
        self.duration = 0.0


class PerformanceEvaluator:
    """
        Callback for valuate the performance of the model.
    Args:
        actor_num_params: The number of parameters of the actor model.
        critic_num_params: The number of parameters of the critic model.
        initial_model_num_params: The number of parameters of the initial model.
        reward_model_num_params: The number of parameters of the reward model.
        enable_grad_checkpoint: Whether to enable gradient checkpointing.
        ignore_episodes: The number of episodes to ignore when calculating the performance.
    """

    def __init__(
        self,
        model_numel: int,
        enable_grad_checkpoint: bool = False,
        ignore_steps: int = 0,
        dp_world_size: Optional[int] = None,
    ) -> None:
        self.model_numel = model_numel
        self.enable_grad_checkpoint = enable_grad_checkpoint
        self.ignore_steps = ignore_steps
        self.dp_world_size = dp_world_size
        self.world_size = dist.get_world_size()
        self.disable: bool = False
        self.timer = Timer()
        self.num_samples: int = 0
        self.flop: int = 0

    def on_step_start(self, step: int) -> None:
        self.disable = self.ignore_steps > 0 and step < self.ignore_steps
        if self.disable:
            return
        torch.cuda.synchronize()
        self.timer.start()

    def on_step_end(self, input_ids: Tensor, **kwargs) -> None:
        if self.disable:
            return
        torch.cuda.synchronize()
        self.timer.end()

        batch_size, seq_len = input_ids.shape

        self.num_samples += batch_size
        self.flop += batch_size * seq_len * self.model_numel * 2 * (3 + int(self.enable_grad_checkpoint))

    def on_fit_end(self) -> None:
        avg_duration = all_reduce_mean(self.timer.duration, self.world_size)
        avg_throughput = self.num_samples * self.dp_world_size / (avg_duration + 1e-12)
        mp_world_size = self.world_size // self.dp_world_size
        avg_tflops_per_gpu = self.flop / 1e12 / (avg_duration + 1e-12) / mp_world_size
        if dist.get_rank() == 0:
            print(
                f"num_samples: {self.num_samples}, dp_world_size: {self.dp_world_size}, flop: {self.flop}, avg_duration: {avg_duration}, "
                f"avg_throughput: {avg_throughput}"
            )
            print(f"Throughput: {avg_throughput:.2f} samples/sec, TFLOPS per GPU: {avg_tflops_per_gpu:.2f}")


if __name__ == "__main__":
    main()
