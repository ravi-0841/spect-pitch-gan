#!/bin/bash -l
#SBATCH --partition=gpuk80
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH -t 20:00:00

module load cuda/10.1
module load singularity

# this works on MARCC, work on Lustre /scratch
cd $HOME/data/ravi/spect-pitch-gan

# pull the image from singularity hub
# singularity pull --name tf_1_12.simg shub://ravi-0841/singularity-tensorflow-1.14

# export singularity home path
export SINGULARITY_HOME=$PWD:/home/$USER

#singularity exec --nv ./tf_1_12.simg python3 main_sum_ec_f0_momenta_wasserstein.py --lambda_cycle_pitch $1 --lambda_cycle_energy $2 --lambda_identity_energy 0.0 --lambda_momenta 1e-06 --generator_learning_rate 1e-05 --discriminator_learning_rate 1e-07 --emotion_pair $3 --tf_random_seed $4 --gender_shuffle $5

singularity exec --nv ./tf_1_12.simg python3 main_joint_ec_f0_momenta_wasserstein.py --lambda_cycle_pitch $1 --lambda_cycle_energy $2 --lambda_identity_energy 0.0 --lambda_momenta 1e-06 --generator_learning_rate 1e-05 --discriminator_learning_rate 1e-07 --emotion_pair $3
