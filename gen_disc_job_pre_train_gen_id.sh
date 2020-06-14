#!/bin/bash -l
#SBATCH --partition=gpup100
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH -t 10:00:00

module load cuda/9.0
module load singularity

# this works on MARCC, work on Lustre /scratch
cd $HOME/data/ravi/spect-pitch-gan

# pull the image from singularity hub
singularity pull --name tf.simg shub://ravi-0841/singularity-tensorflow-1.14

# export singularity home path
export SINGULARITY_HOME=$PWD:/home/$USER

singularity exec --nv ./tf.simg python3 main_pre_train_gen_id.py --lambda_cycle_pitch $1 --lambda_cycle_mfc $2 --lambda_identity_mfc $3 --lambda_momenta $4 --generator_learning_rate $5 --discriminator_learning_rate $6
