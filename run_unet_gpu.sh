#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --job-name unet_4gpu
#SBATCH --time 03:00:00

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --ntasks-per-socket=2
#SBATCH --mem-per-cpu=20G
#SBATCH --gres=gpu:4

#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --mail-user=tmakinen@princeton.edu

# load modules
module load anaconda3


# Run sim_format
#python sim_format.py

# Add foreground to cutouts
#python add_fg_cosmo.py

# Composite data together
#python composite_data.py

# Run PCA analysis (3 components)
#python pca_format.py

# LATER: run unet analysis
conda activate tf_gpu
python remove_foreground_main_gpu.py