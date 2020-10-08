#!/bin/bash -l
#SBATCH --time=00:40:00
#SBATCH --partition=shared
#SBATCH --mem=32gb
#SBATCH --cpus-per-task=1
module load matlab/R2018a

cd $HOME/data/ravi/pitch-lddmm-spect

matlab  -nodisplay -nosplash -r "generate_mfc_momenta $1 $2 $3;"
echo "matlab exit code: $?"
