#!/bin/bash -l
#SBATCH --partition=gpuk80
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH -t 48:00:00

module load cuda/10.1
module load singularity

# this works on MARCC, work on Lustre /scratch
cd $HOME/data/ravi/spect-pitch-gan

# pull the image from singularity hub
singularity pull --name tf_1_12.simg shub://ravi-0841/singularity-tensorflow-1.14

# export singularity home path
export SINGULARITY_HOME=$PWD:/home/$USER

singularity exec --nv ./tf_1_12.simg python3 main_supervised_energy_f0_momenta_wasserstein.py --lambda_energy $1 --lambda_momenta_pitch $2 --lambda_momenta_energy $3 --generator_learning_rate $4 --discriminator_learning_rate 1e-06
