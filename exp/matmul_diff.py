import time

import torch

from colossalai.moe.experts import MLPExperts


def benchmark(model, inputs, warmup=10, runs=5):
    for _ in range(warmup):
        output = model(inputs)
        output.sum().backward()

    torch.cuda.synchronize()
    time0 = time.time()
    for _ in range(runs):
        output = model(inputs)
        output.sum().backward()
    torch.cuda.synchronize()
    time1 = time.time()
    return (time1 - time0) / runs


def matmul_diff():
    num_experts = 2
    hidden_size = 4096
    intermediate_size = 14336

    experts = MLPExperts(
        num_experts=num_experts,
        expert_parallel=None,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        activation="silu",
        gated=True,
        use_kernel=False,
    )
    experts = experts.half().cuda()

    max_seq_len = 4096
    batch_size = 4
    interval = 512

    for seq_len in range(interval, max_seq_len + interval, interval):
        inputs = torch.randn(num_experts, batch_size * seq_len, 4096, device="cuda", dtype=torch.half)
        time = benchmark(experts, inputs)
        print(f"seq_len: {seq_len}, time: {time:.6f} s")


if __name__ == "__main__":
    matmul_diff()
