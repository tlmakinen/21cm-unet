#!/bin/bash
#SBATCH -p gpu
#SBATCH -N1
#SBATCH --cpus-per-task=16
#SBATCH --time=0-08:30:00
#SBATCH --job-name="network1-fullrange"
#SBATCH --gres=gpu:v100-32gb:2
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --mail-user=tmakinen@princeton.edu
#SBATCH --output network1-%J.log
module load  gcc/7.4.0 cuda/10.1.243_418.87.00 cudnn/v7.6.2-cuda-10.1 nccl/2.4.2-cuda-10.1 python3/3.7.3

#python3 clean_n2.py

python3 big_gen_cca.py
