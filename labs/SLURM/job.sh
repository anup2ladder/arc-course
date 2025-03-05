#!/bin/bash
#Set job requirements
#SBATCH -p gpu_a100
#SBATCH --gpus=1
#SBATCH -n 18
#SBATCH -t 5:00

conda deactivate
module load 2024
module load Anaconda3/2024.06-1
conda activate arccourse

# At this point I was getting the following error:
# ImportError: /lib64/libstdc++.so.6: version `GLIBCXX_3.4.30' not found (required by /home/carcassi/.conda/envs/arccourse/lib/python3.11/site-packages/zmq/backend/cython/../../../../../libzmq.so.5)
# This can be fixed loading the compiler:
module load GCC/13.3.0

# This is because otherwise UNSLOTH does not recognize the name of the GPU
# so we refer to it as cuda:0
export CUDA_VISIBLE_DEVICES=0

python GRPO_train.py
