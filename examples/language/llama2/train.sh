CUDA_VISIBLE_DEVICES=3 torchrun --nproc_per_node 1 --standalone pretrain.py -p zero2 -b 32 -c base --lr 3e-4 -l 1024 -a
