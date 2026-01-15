#!/bin/bash
#SBATCH --job-name=large_cosmic_pop     # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks-per-node=2     # number of tasks per node
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=16G             # memory per cpu-core (4G is default)
#SBATCH --time=4:00:00          # total run time limit (HH:MM:SS)  HAD -r 0.0001
#SBATCH -p conroy,itc_cluster,sapphire,shared #partition to submit to
#SBATCH --mail-type=END


module purge
module load python/3.12.5-fasrc01
module load gcc/14.2.0-fasrc01 gsl/2.8-fasrc01
mamba activate cosmic_x_amuse

python -u ~/p26/scripts/gen_cosmic_pop.py --n_stars=1000000 --cosmic_samples=1000000 --f="large_COSMIC_stellar_pop"
