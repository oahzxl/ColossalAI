device="0"

CUDA_VISIBLE_DEVICES=$device torchrun --nproc_per_node 1 --standalone pretrain.py --plugin zero2 --batch_size 32 --config base --lr 3e-4 --max_length 1024 --flash_attention --warmup_steps 2000 --mixed_precision bf16 --save_interval 10000
CUDA_VISIBLE_DEVICES=$device torchrun --nproc_per_node 1 --standalone pretrain.py --plugin zero2 --batch_size 32 --config base --lr 1e-4 --max_length 1024 --flash_attention --warmup_steps 2000 --mixed_precision bf16 --save_interval 10000
CUDA_VISIBLE_DEVICES=$device torchrun --nproc_per_node 1 --standalone pretrain.py --plugin zero2 --batch_size 32 --config base --lr 1e-3 --max_length 1024 --flash_attention --warmup_steps 2000 --mixed_precision bf16 --save_interval 10000
