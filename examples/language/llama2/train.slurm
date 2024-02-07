#!/bin/bash

#SBATCH -J vmoe           # Job name
#SBATCH -o vmoe.o         # Name of stdout output file
#SBATCH -e vmoe.e         # Name of stderr error file
#SBATCH -p gpu-a100       # Queue name
#SBATCH -N 4              # Node num
#SBATCH -n 12             # Task num
#SBATCH -t 48:00:00       # Time hh:mm:ss
#SBATCH --mail-type=all   # Send email at begin and end of job
#SBATCH --mail-user=xuanlei@comp.nus.edu.sg

export NCCL_NET_GDR_LEVEL=SYS
export NCCL_NET_GDR_READ=1

ibrun true

module load cuda/12.0

# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/home1/09877/xuanlei/miniconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/home1/09877/xuanlei/miniconda3/etc/profile.d/conda.sh" ]; then
        . "/home1/09877/xuanlei/miniconda3/etc/profile.d/conda.sh"
    else
        export PATH="/home1/09877/xuanlei/miniconda3/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<
conda activate v-moe

HOSTS=$(scontrol show hostnames $SLURM_NODELIST | sed -z 's/\n/,/g;s/,$//')
MASTER=$(echo $HOSTS | cut -d, -f1)
echo "master:$MASTER, hosts:$HOSTS"

colossalai run --nproc_per_node 3 --num_nodes $SLURM_JOB_NUM_NODES --host $HOSTS --master_addr $MASTER pretrain.py --plugin zero2 --batch_size 24 --config base --lr 3e-4 --max_length 1024 --warmup_steps 2000 --mixed_precision bf16 --save_interval 1000 --flash_attention --load /work/09877/xuanlei/ls6/VMoE/examples/language/llama2/checkpoint/2024-02-06-16-02-04/epoch0-step10000