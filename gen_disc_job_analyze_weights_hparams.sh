#!/bin/bash -l
#SBATCH --partition=gpuk80
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH -t 23:00:00

module load cuda/9.0
module load singularity

# this works on MARCC, work on Lustre /scratch
cd $HOME/data/ravi/spect-pitch-gan

# pull the image from singularity hub
singularity pull --name tf.simg shub://ravi-0841/singularity-tensorflow-1.14

# export singularity home path
export SINGULARITY_HOME=$PWD:/home/$USER

singularity exec --nv ./tf.simg python3 main_analyze_weights.py --generator_learning_rate $1 --discriminator_learning_rate $2 --lambda_cycle_pitch $3 --lambda_cycle_mfc $4
