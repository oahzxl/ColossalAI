#!/bin/bash

set -xue

NUM_GPU=8
MODEL="test"
SEQ_LENGTH=512
WARMUP=5
ACTIVE=2


# hybrid
torchrun --standalone --nproc_per_node $NUM_GPU \
    benchmark.py \
    --model_name $MODEL \
    --batch_size 2 \
    --max_length $SEQ_LENGTH \
    --warmup $WARMUP \
    --active $ACTIVE \
    --plugin ep_zero \
    --extra_dp_size 2 \
    --zero_stage 2 \
    --half_local_world_size \
    --enable_dp_balance \
    --enable_tp_balance
