import argparse
import os

import torch
import torch.distributed as dist
from colossal_moe.models.mixtral_checkpoint import MixtralMoECheckpointIO
from colossal_moe.models.mixtral_layer import replace_moe_layer
from colossal_moe.models.mixtral_policy import MixtralForCausalLMPolicy
from colossal_moe.utils import move_to_cuda
from tqdm import tqdm
from transformers.models.mixtral import MixtralConfig, MixtralForCausalLM
from utils import PerformanceEvaluator, RandomDataset, get_global_loss, get_model_numel

import colossalai
from colossalai.booster import Booster
from colossalai.booster.plugin.moe_hybrid_parallel_plugin import MoeHybridParallelPlugin
from colossalai.cluster import DistCoordinator
from colossalai.moe import MOE_MANAGER, apply_load_balance
from colossalai.nn.optimizer import HybridAdam
from colossalai.utils import get_current_device


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


if __name__ == "__main__":
    main()
