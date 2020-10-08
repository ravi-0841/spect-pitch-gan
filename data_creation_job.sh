#!/bin/bash -l
#SBATCH --time=00:40:00
#SBATCH --partition=shared
#SBATCH --mem=32gb
#SBATCH --cpus-per-task=1
module load matlab/R2018a

cd $HOME/data/ravi/spect-pitch-gan

matlab  -nodisplay -nosplash -r "generate_momenta $SLURM_ARRAY_TASK_ID cmu-arctic train;"
echo "matlab exit code: $?"
