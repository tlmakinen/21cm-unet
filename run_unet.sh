#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --time 02:00:00
#SBATCH --job-name clean_data


# get tunneling info
module load anaconda3
conda activate 21cm




# Run sim_format
python sim_format.py

# Add foreground to cutouts
python add_fg_cosmo.py

# Composite data together
python composite_data.py

# Run PCA analysis (3 components)
python pca_format.py

# LATER: run unet analysis