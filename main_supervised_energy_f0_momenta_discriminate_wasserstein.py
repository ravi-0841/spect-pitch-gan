import os
import numpy as np
import argparse
import time
import librosa
import sys
import scipy.io.wavfile as scwav
import scipy.io as scio
import scipy.signal as scisig
import pylab
import logging

from glob import glob
from nn_models.model_supervised_energy_f0_momenta_discriminate_wasserstein import VariationalCycleGAN
from utils.helper import smooth, generate_interpolation
import utils.preprocess as preproc
from importlib import reload
from encoder_decoder import AE


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def train(train_dir, model_dir, model_name, random_seed, \
            tensorboard_log_dir, pre_train=None, \
            lambda_energy=0, lambda_momenta_pitch=0, 
            lambda_momenta_energy=0, generator_learning_rate=1e-05, 
            discriminator_learning_rate=1e-05):

    np.random.seed(random_seed)

    num_epochs = 1000
    mini_batch_size = 1 # mini_batch_size = 1 is better

    sampling_rate = 16000
    num_mcep = 23
    frame_period = 5
    n_frames = 128

    lc_lm = "le_"+str(lambda_energy)+'_lme_'+str(lambda_momenta_energy) \
            +'_lmp_'+str(lambda_momenta_pitch)+'_supervised_mwd_spect_male_female'

    model_dir = os.path.join(model_dir, lc_lm)

    logger_file = './log/'+lc_lm+'.log'
    if os.path.exists(logger_file):
        os.remove(logger_file)

    reload(logging)
    logging.basicConfig(filename=logger_file, \
                            level=logging.DEBUG)

    print("lambda_energy - {}".format(lambda_energy))
    print("lambda_momenta_pitch - {}".format(lambda_momenta_pitch))
    print("lambda_momenta_energy - {}".format(lambda_momenta_energy))

    logging.info("lambda_energy - {}".format(lambda_energy))
    logging.info("lambda_momenta_pitch - {}".format(lambda_momenta_pitch))
    logging.info("lambda_momenta_energy - {}".format(lambda_momenta_energy))
    logging.info("generator_lr - {}".format(generator_learning_rate))
    logging.info("discriminator_lr - {}".format(discriminator_learning_rate))

    if not os.path.isdir("./pitch_spect/"+lc_lm):
        os.makedirs(os.path.join("./pitch_spect/", lc_lm))
    else:
        for f in glob(os.path.join("./pitch_spect/", \
                lc_lm, "*.png")):
            os.remove(f)
    
    print('Preprocessing Data...')

    start_time = time.time()

    data_train = scio.loadmat(os.path.join(train_dir, 'spect_energy_cmu_arctic.mat'))

    pitch_A_train = data_train['src_f0_feat']
    pitch_B_train = data_train['tar_f0_feat']
    energy_A_train = np.log(data_train['src_ec_feat'] + 1e-06)
    energy_B_train = np.log(data_train['tar_ec_feat'] + 1e-06)
    mfc_A_train = data_train['src_mfc_feat']
    mfc_B_train = data_train['tar_mfc_feat']
    momenta_A2B_f0 = data_train['momenta_f0_A2B']
    momenta_B2A_f0 = data_train['momenta_f0_B2A']
    momenta_A2B_ec = data_train['momenta_ec_A2B']
    momenta_B2A_ec = data_train['momenta_ec_B2A']


    # Randomly shuffle the trainig data
    indices_train = np.arange(0, pitch_A_train.shape[0])
    np.random.shuffle(indices_train)
    pitch_A_train = pitch_A_train[indices_train]
    energy_A_train = energy_A_train[indices_train]
    mfc_A_train = mfc_A_train[indices_train]
    momenta_A2B_f0 = momenta_A2B_f0[indices_train]
    momenta_A2B_ec = momenta_A2B_ec[indices_train]
    pitch_B_train = pitch_B_train[indices_train]
    energy_B_train = energy_B_train[indices_train]
    mfc_B_train = mfc_B_train[indices_train]
    momenta_B2A_f0 = momenta_B2A_f0[indices_train]
    momenta_B2A_ec = momenta_B2A_ec[indices_train]

    end_time = time.time()
    time_elapsed = end_time - start_time

    print('Preprocessing Done.')

    print('Time Elapsed for Data Preprocessing: %02d:%02d:%02d' % (time_elapsed // 3600, \
                                                                   (time_elapsed % 3600 // 60), \
                                                                   (time_elapsed % 60 // 1)))
    
    #use pre_train arg to provide trained model
    model = VariationalCycleGAN(dim_pitch=1, dim_energy=1, dim_mfc=23, n_frames=n_frames, 
                                pre_train=pre_train, log_file_name=lc_lm)
    
    for epoch in range(1,num_epochs+1):

        print('Epoch: %d' % epoch)
        logging.info('Epoch: %d' % epoch)

        start_time_epoch = time.time()

        mfc_A, pitch_A, energy_A, momenta_pitch_A2B, momenta_energy_A2B, \
            mfc_B, pitch_B, energy_B, momenta_pitch_B2A, momenta_energy_B2A \
                = preproc.sample_data_energy_momenta(mfc_A=mfc_A_train, 
                    mfc_B=mfc_B_train, pitch_A=pitch_A_train, pitch_B=pitch_B_train, 
                    energy_A=energy_A_train, energy_B=energy_B_train, 
                    momenta_pitch_A=momenta_A2B_f0, momenta_pitch_B=momenta_B2A_f0, 
                    momenta_energy_A=momenta_A2B_ec, momenta_energy_B=momenta_B2A_ec)
        
        n_samples = energy_A.shape[0]
        
        train_gen_loss = []
        train_pitch_loss = []
        train_energy_loss = []

        for i in range(n_samples // mini_batch_size):

            start = i * mini_batch_size
            end = (i + 1) * mini_batch_size

            generator_loss, pitch_loss, energy_loss, \
            gen_pitch_A, gen_energy_A, gen_pitch_B, \
            gen_energy_B, mom_pitch_A, mom_pitch_B, \
            mom_energy_A, mom_energy_B \
                = model.train(mfc_A=mfc_A[start:end], energy_A=energy_A[start:end], 
                    pitch_A=pitch_A[start:end], mfc_B=mfc_B[start:end],  
                    energy_B=energy_B[start:end], pitch_B=pitch_B[start:end], 
                    lambda_energy=lambda_energy, lambda_momenta_pitch=lambda_momenta_pitch, 
                    lambda_momenta_energy=lambda_momenta_energy, 
                    momenta_pitch_A2B=momenta_pitch_A2B, 
                    momenta_energy_A2B=momenta_energy_A2B, 
                    momenta_pitch_B2A=momenta_pitch_B2A, 
                    momenta_energy_B2A=momenta_energy_B2A, 
                    generator_learning_rate=generator_learning_rate, 
                    discriminator_learning_rate=discriminator_learning_rate)
            
            train_gen_loss.append(generator_loss)
            train_pitch_loss.append(pitch_loss)
            train_energy_loss.append(energy_loss)

        
        print("Train Generator Loss- {}".format(np.mean(train_gen_loss)))
        print("Train Pitch Loss- {}".format(np.mean(train_pitch_loss)))
        print("Train Energy Loss- {}".format(np.mean(train_energy_loss)))
        
        logging.info("Train Generator Loss- {}".format(np.mean(train_gen_loss)))
        logging.info("Train Pitch Loss- {}".format(np.mean(train_pitch_loss)))
        logging.info("Train Energy Loss- {}".format(np.mean(train_energy_loss)))
        
        end_time_epoch = time.time()
        time_elapsed_epoch = end_time_epoch - start_time_epoch

        print('Time Elapsed for This Epoch: %02d:%02d:%02d' % (time_elapsed_epoch // 3600, \
                (time_elapsed_epoch % 3600 // 60), (time_elapsed_epoch % 60 // 1)))

        logging.info('Time Elapsed for This Epoch: %02d:%02d:%02d' % (time_elapsed_epoch // 3600, \
                (time_elapsed_epoch % 3600 // 60), (time_elapsed_epoch % 60 // 1)))

        if epoch % 50 == 0:
            
            cur_model_name = model_name+"_"+str(epoch)+".ckpt"
            model.save(directory=model_dir, filename=cur_model_name)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description = 'Train VariationalCycleGAN model for datasets.')

    emo_dict = {"neu-ang":['neutral', 'angry'], \
                        "neu-sad":['neutral', 'sad'], \
                        "neu-hap":['neutral', 'happy']}

    emo_pair = "cmu-arctic"
    train_dir_default = "./data/"+emo_pair
    model_dir_default = "./model/"+emo_pair
    model_name_default = emo_pair
    tensorboard_log_dir_default = './log/'+emo_pair
    random_seed_default = 0

    parser.add_argument('--train_dir', type=str, help='Directory for A.', 
            default=train_dir_default)
    parser.add_argument('--model_dir', type=str, help='Directory for saving models.', 
            default=model_dir_default)
    parser.add_argument('--model_name', type=str, help='File name for saving model.', 
            default=model_name_default)
    parser.add_argument('--random_seed', type=int, help='Random seed for model training.', 
            default=random_seed_default)
    parser.add_argument('--tensorboard_log_dir', type=str, help='TensorBoard log directory.', 
            default=tensorboard_log_dir_default)
    parser.add_argument('--current_iter', type=int, help="Current iteration of the model (Fine tuning)", 
            default=1)
    parser.add_argument('--lambda_energy', type=float, help="hyperparam for loss energy", 
            default=10.0)
    parser.add_argument('--lambda_momenta_energy', type=float, help="hyperparam for momenta energy", 
            default=10.0)
    parser.add_argument('--lambda_momenta_pitch', type=float, help="hyperparam for momenta pitch", 
            default=10.0)
    parser.add_argument('--generator_learning_rate', type=float, help="generator learning rate", 
            default=1e-06)
    parser.add_argument('--discriminator_learning_rate', type=float, help="discriminator learning rate", 
            default=1e-07)
    
    argv = parser.parse_args()

    train_dir = argv.train_dir
    model_dir = argv.model_dir
    model_name = argv.model_name
    random_seed = argv.random_seed
    tensorboard_log_dir = argv.tensorboard_log_dir

    lambda_energy = argv.lambda_energy
    lambda_momenta_energy = argv.lambda_momenta_energy
    lambda_momenta_pitch = argv.lambda_momenta_pitch

    generator_learning_rate = argv.generator_learning_rate
    discriminator_learning_rate = argv.discriminator_learning_rate

    train(train_dir=train_dir, model_dir=model_dir, model_name=model_name, 
          random_seed=random_seed, tensorboard_log_dir=tensorboard_log_dir, 
          pre_train=None, lambda_energy=lambda_energy, 
          lambda_momenta_pitch=lambda_momenta_pitch, 
          lambda_momenta_energy=lambda_momenta_energy, 
          generator_learning_rate=generator_learning_rate, 
          discriminator_learning_rate=discriminator_learning_rate)
