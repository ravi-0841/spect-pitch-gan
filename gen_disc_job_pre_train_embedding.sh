#!/bin/bash -l
#SBATCH --partition=gpuk80
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH -t 14:00:00

module load cuda/10.1
module load singularity

# this works on MARCC, work on Lustre /scratch
cd $HOME/data/ravi/spect-pitch-gan

# pull the image from singularity hub
singularity pull --name tf_1_12.simg shub://ravi-0841/singularity-tensorflow-1.14

# export singularity home path
export SINGULARITY_HOME=$PWD:/home/$USER

singularity exec --nv ./tf_1_12.simg python3 main_pre_train_embedding.py --lambda_cycle_pitch $1 --lambda_cycle_mfc $2 --lambda_identity_mfc $3 --lambda_momenta $4 --generator_learning_rate $5 --discriminator_learning_rate $6
