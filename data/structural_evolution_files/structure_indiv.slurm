#!/bin/bash
#SBATCH --job-name=struc10_new
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH -p conroy,itc_cluster,shared,sapphire
#SBATCH --output=slurm_%j.out


module load python/3.12.5-fasrc01 gcc/14.2.0-fasrc01 gsl/2.8-fasrc01
mamba activate petar_env

python -u ~/binaries_in_streams/scripts/structural_evolution_NEW.py \
    --sim_n=10 --start=0 --stop=1500 --step=1