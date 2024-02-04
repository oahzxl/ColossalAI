import argparse
import os
import random
import resource
import time
from contextlib import nullcontext
from functools import partial
from typing import Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from data_utils import load_json, prepare_dataloader, save_json
from datasets import load_dataset
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import AutoTokenizer
from transformers.models.mixtral import MixtralConfig, MixtralForCausalLM

import colossalai
from colossalai.booster import Booster
from colossalai.booster.plugin import GeminiPlugin, HybridParallelPlugin, LowLevelZeroPlugin
from colossalai.cluster import DistCoordinator
from colossalai.lazy import LazyInitContext
from colossalai.nn.lr_scheduler import CosineAnnealingWarmupLR
from colossalai.nn.optimizer import HybridAdam
from colossalai.utils import get_current_device


def seed_everything(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_model_numel(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def format_numel_str(numel: int) -> str:
    B = 1024**3
    M = 1024**2
    K = 1024
    if numel >= B:
        return f"{numel / B:.2f} B"
    elif numel >= M:
        return f"{numel / M:.2f} M"
    elif numel >= K:
        return f"{numel / K:.2f} K"
    else:
        return f"{numel}"


def format_token_str(numel: int) -> str:
    B = 1000**3
    M = 1000**2
    K = 1000
    if numel >= B:
        return f"{numel / B:.2f}B"
    elif numel >= M:
        return f"{numel / M:.2f}M"
    elif numel >= K:
        return f"{numel / K:.2f}K"
    else:
        return f"{numel}"


def args_to_dict(args):
    return {k: v for k, v in vars(args).items() if not k.startswith("_")}


def tokenize_batch_for_pretrain(batch, tokenizer: Optional[AutoTokenizer] = None, max_length: int = 2048):
    texts = [sample["text"] for sample in batch]
    data = tokenizer(texts, return_tensors="pt", padding="max_length", truncation=True, max_length=max_length)
    data["labels"] = data["input_ids"].clone()
    return data


def all_reduce_mean(tensor: torch.Tensor) -> torch.Tensor:
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    tensor = tensor.data
    tensor.div_(dist.get_world_size())
    return tensor


def save(
    booster: Booster,
    model: nn.Module,
    optimizer: Optimizer,
    lr_scheduler: _LRScheduler,
    epoch: int,
    step: int,
    token_num: int,
    batch_size: int,
    coordinator: DistCoordinator,
    save_dir: str,
):
    save_dir = os.path.join(save_dir, f"epoch{epoch}-step{step}")
    os.makedirs(os.path.join(save_dir, "model"), exist_ok=True)

    booster.save_model(model, os.path.join(save_dir, "model"), shard=True)
    booster.save_optimizer(optimizer, os.path.join(save_dir, "optimizer"), shard=True)
    booster.save_lr_scheduler(lr_scheduler, os.path.join(save_dir, "lr_scheduler"))
    running_states = {
        "epoch": epoch,
        "step": step,
        "sample_start_index": step * batch_size,
        "token_num": token_num,
    }
    if coordinator.is_master():
        save_json(running_states, os.path.join(save_dir, "running_states.json"))


def load(
    booster: Booster, model: nn.Module, optimizer: Optimizer, lr_scheduler: _LRScheduler, load_dir: str
) -> Tuple[int, int, int]:
    booster.load_model(model, os.path.join(load_dir, "model"))
    booster.load_optimizer(optimizer, os.path.join(load_dir, "optimizer"))
    booster.load_lr_scheduler(lr_scheduler, os.path.join(load_dir, "lr_scheduler"))
    running_states = load_json(os.path.join(load_dir, "running_states.json"))
    return (
        running_states["epoch"],
        running_states["step"],
        running_states["sample_start_index"],
        running_states["token_num"],
    )


def _criterion(outputs, inputs):
    return outputs.loss


def main():
    # ==============================
    # Parse Arguments
    # ==============================
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, default="7b", help="Model configuration")
    parser.add_argument(
        "-p",
        "--plugin",
        choices=["gemini", "gemini_auto", "zero2", "zero2_cpu", "hybrid_parallel"],
        default="gemini",
        help="Choose which plugin to use",
    )
    parser.add_argument(
        "-d", "--dataset", type=str, default="togethercomputer/RedPajama-Data-1T-Sample", help="Data set path"
    )
    parser.add_argument("-e", "--num_epochs", type=int, default=1, help="Number of epochs")
    parser.add_argument("-b", "--batch_size", type=int, default=2, help="Local batch size")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("-w", "--weigth_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("-s", "--warmup_steps", type=int, default=2000, help="Warmup steps")
    parser.add_argument("-g", "--grad_checkpoint", action="store_true", help="Use gradient checkpointing")
    parser.add_argument("-l", "--max_length", type=int, default=4096, help="Max sequence length")
    parser.add_argument("-x", "--mixed_precision", default="fp16", choices=["fp16", "bf16"], help="Mixed precision")
    parser.add_argument("-i", "--save_interval", type=int, default=1000, help="Save interval")
    parser.add_argument("-o", "--save_dir", type=str, default="checkpoint", help="Checkpoint directory")
    parser.add_argument("-f", "--load", type=str, default=None, help="Load checkpoint")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="Gradient clipping")
    parser.add_argument("-t", "--tensorboard_dir", type=str, default="tb_logs", help="Tensorboard directory")
    parser.add_argument("-a", "--flash_attention", action="store_true", help="Use Flash Attention")
    args = parser.parse_args()

    # ==============================
    # Initialize Distributed Training
    # ==============================
    colossalai.launch_from_torch({})
    seed_everything(42)
    coordinator = DistCoordinator()

    # ==============================
    # Initialize Model Configs
    # ==============================
    MODEL_CONFIGS = {
        # follow opt-125m
        "base": MixtralConfig(
            hidden_size=768,
            intermediate_size=3072,
            num_hidden_layers=12,
            num_attention_heads=12,
            num_key_value_heads=4,
            max_position_embeddings=args.max_length,
            num_local_experts=8,
            use_cache=False,
            _attn_implementation="flash_attention_2" if args.flash_attention else "eager",
        ),
    }

    # ==============================
    # Initialize Booster
    # ==============================
    if args.plugin == "gemini":
        plugin = GeminiPlugin(precision=args.mixed_precision, initial_scale=2**16, max_norm=args.grad_clip)
    elif args.plugin == "gemini_auto":
        plugin = GeminiPlugin(
            precision=args.mixed_precision, placement_policy="auto", initial_scale=2**16, max_norm=args.grad_clip
        )
    elif args.plugin == "zero2":
        plugin = LowLevelZeroPlugin(
            stage=2, precision=args.mixed_precision, initial_scale=2**16, max_norm=args.grad_clip
        )
    elif args.plugin == "zero2_cpu":
        plugin = LowLevelZeroPlugin(
            stage=2, precision=args.mixed_precision, initial_scale=2**16, cpu_offload=True, max_norm=args.grad_clip
        )
    elif args.plugin == "hybrid_parallel":
        # modify the param accordingly, default configuration is for llama2-7b
        plugin = HybridParallelPlugin(
            tp_size=4,
            pp_size=2,
            num_microbatches=None,
            microbatch_size=1,
            enable_jit_fused=False,
            zero_stage=0,
            precision="fp32",
            initial_scale=1,
        )
    else:
        raise ValueError(f"Unknown plugin {args.plugin}")

    booster = Booster(plugin=plugin)

    use_pipeline = isinstance(booster.plugin, HybridParallelPlugin) and booster.plugin.pp_size > 1
    is_pp_last_stage = use_pipeline and booster.plugin.stage_manager.is_last_stage()
    print_flag = (not use_pipeline and coordinator.is_master()) or (use_pipeline and is_pp_last_stage)

    # ==============================
    # Initialize Tensorboard
    # ==============================
    time_prefix = time.strftime("%Y-%m-%d-%H-%M-%S")
    if print_flag:
        tb_dir = os.path.join(args.tensorboard_dir, time_prefix)
        os.makedirs(tb_dir, exist_ok=True)
        writer = SummaryWriter(tb_dir)
        writer.add_text("args: ", str(args_to_dict(args)))

    # ==============================
    # Initialize Tokenizer, Dataset and Dataloader
    # ==============================
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mixtral-8x7B-v0.1")
    # follows fast chat: https://github.com/lm-sys/FastChat/blob/main/fastchat/train/train.py#L257
    tokenizer.pad_token = tokenizer.unk_token

    train_ds = load_dataset(
        "/data/personal/nus-zxl/VerticalMoE/data_prepare/redpajama_dataset.py",
        "c4_1-3",
        trust_remote_code=True,
        split="train",
    )
    dataloader = prepare_dataloader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        collate_fn=partial(tokenize_batch_for_pretrain, tokenizer=tokenizer, max_length=args.max_length),
        num_workers=8,
        prefetch_factor=1,
    )
    total_token_num = 60000000000  # 60B

    # ==============================
    # Initialize Model, Optimizer and LR Scheduler
    # ==============================
    config = MODEL_CONFIGS[args.config]
    # use lazy init when using GeminiPlugin
    init_ctx = (
        LazyInitContext(default_device=get_current_device()) if isinstance(plugin, GeminiPlugin) else nullcontext()
    )

    with init_ctx:
        default_dtype = torch.float16 if args.mixed_precision == "fp16" else torch.bfloat16
        torch.set_default_dtype(default_dtype)
        model = MixtralForCausalLM(config)
        torch.set_default_dtype(torch.float)

    if args.grad_checkpoint:
        model.gradient_checkpointing_enable()
    # if args.flash_attention:
    #     assert SUPPORT_XFORMERS, "Use flash attention while xfomers is not installed"
    #     replace_xformers(model)

    model_numel = get_model_numel(model)
    coordinator.print_on_master(f"Model params: {format_numel_str(model_numel)}")

    optimizer = HybridAdam(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=args.weigth_decay)
    lr_scheduler = CosineAnnealingWarmupLR(
        optimizer, total_steps=args.num_epochs * len(dataloader), warmup_steps=args.warmup_steps, eta_min=0.1 * args.lr
    )
    default_dtype = torch.float16 if args.mixed_precision == "fp16" else torch.bfloat16
    torch.set_default_dtype(default_dtype)
    model, optimizer, _, dataloader, lr_scheduler = booster.boost(
        model, optimizer, dataloader=dataloader, lr_scheduler=lr_scheduler
    )
    torch.set_default_dtype(torch.float)

    coordinator.print_on_master(f"Booster init max CUDA memory: {torch.cuda.max_memory_allocated()/1024**2:.2f} MB")
    coordinator.print_on_master(
        f"Booster init max CPU memory: {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024:.2f} MB"
    )

    # load checkpoint if specified
    start_epoch = 0
    start_step = 0
    sampler_start_idx = 0
    token_num = 0

    if args.load is not None:
        coordinator.print_on_master("Loading checkpoint")
        start_epoch, start_step, sampler_start_idx, token_num = load(booster, model, optimizer, lr_scheduler, args.load)
        coordinator.print_on_master(
            f"Loaded checkpoint {args.load} at epoch {start_epoch} step {start_step}, token_num {token_num}"
        )

    num_steps_per_epoch = len(dataloader)

    # if resume training, set the sampler start index to the correct value
    dataloader.sampler.set_start_index(sampler_start_idx)
    for epoch in range(start_epoch, args.num_epochs):
        dataloader.sampler.set_epoch(epoch)
        dataloader_iter = iter(dataloader)

        with tqdm(
            range(start_step, num_steps_per_epoch),
            desc=f"Epoch {epoch}",
            disable=not print_flag,
            total=num_steps_per_epoch,
            initial=start_step,
        ) as pbar:
            for step in pbar:
                if use_pipeline:
                    outputs = booster.execute_pipeline(
                        dataloader_iter, model, _criterion, optimizer, return_loss=True, return_outputs=True
                    )
                    loss = outputs["loss"]
                else:
                    batch = next(dataloader_iter)
                    batch = {k: v.cuda() for k, v in batch.items()}
                    with torch.no_grad():
                        cur_token = (batch["input_ids"] != 0).sum().item()
                        token_num += cur_token
                        if token_num > total_token_num:
                            break
                    outputs = model(**batch)
                    loss = outputs[0]
                    booster.backward(loss, optimizer)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                if not use_pipeline:
                    all_reduce_mean(loss)
                if print_flag:
                    pbar.set_postfix({"token_num": format_token_str(token_num), "loss": loss.item()})
                    writer.add_scalar("loss_per_step", loss.item(), step)
                    writer.add_scalar("loss_per_token", loss.item(), token_num)

                if args.save_interval > 0 and (step + 1) % args.save_interval == 0:
                    coordinator.print_on_master(f"Saving checkpoint")
                    save(
                        booster,
                        model,
                        optimizer,
                        lr_scheduler,
                        epoch,
                        step + 1,
                        token_num,
                        args.batch_size,
                        coordinator,
                        os.path.join(args.save_dir, time_prefix),
                    )
                    coordinator.print_on_master(f"Saved checkpoint at epoch {epoch} step {step + 1}")
        # the continue epochs are not resumed, so we need to reset the sampler start index and start step
        dataloader.sampler.set_start_index(0)
        start_step = 0

    coordinator.print_on_master(f"Max CUDA memory usage: {torch.cuda.max_memory_allocated()/1024**2:.2f} MB")


if __name__ == "__main__":
    main()
