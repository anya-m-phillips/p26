#!/bin/bash
#SBATCH --job-name=structural_evolution_processing
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH -p conroy,itc_cluster,shared,sapphire
#SBATCH --mail-type=END
#SBATCH --output=slurm_n-%A_%a.out
#SBATCH --array=0-23

module load python/3.12.5-fasrc01 gcc/14.2.0-fasrc01 gsl/2.8-fasrc01
mamba activate petar_env

# List of sim_n values to run
SIM_N_VALUES=(0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23)

# List of sim_n values that should use stop=10000
SHORT_STOP_N=(3 5 6 7 9 11 13 14 15 16 17 18 19 20 21 22 23)

# Get sim_n for this array task
SIM_N=${SIM_N_VALUES[$SLURM_ARRAY_TASK_ID]}

# Determine stop value
if [ "$SIM_N" -eq 12 ]; then
    STOP=12325
else
    STOP=15000
    for N in "${SHORT_STOP_N[@]}"; do
        if [ "$SIM_N" -eq "$N" ]; then
            STOP=10000
            break
        fi
    done
fi
# Run the Python script
python -u ~/binaries_in_streams/scripts/structural_evolution_NEW.py \
    --sim_n=$SIM_N --start=0 --stop=$STOP --step=1
