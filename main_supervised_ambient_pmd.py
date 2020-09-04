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

import utils.preprocess as preproc

from glob import glob
from nn_models.model_supervised_pitch_mfc_discriminate import VariationalCycleGAN
from utils.helper import smooth, generate_interpolation
from importlib import reload


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def train(train_dir, model_dir, model_name, random_seed, tensorboard_log_dir, 
        pre_train=None, lambda_pitch=0, lambda_mfc=0, lambda_momenta=0):

    np.random.seed(random_seed)

    num_epochs = 1000
    mini_batch_size = 1 # mini_batch_size = 1 is better

    generator_learning_rate = 0.00001

    sampling_rate = 16000
    num_mcep = 23
    frame_period = 5
    n_frames = 128

    lc_lm = "lp_"+str(lambda_pitch) \
            + '_lm_'+str(lambda_mfc) \
            +"_lmo_"+str(lambda_momenta) + '_supervised_ambient_pmd'

    model_dir = os.path.join(model_dir, lc_lm)

    logger_file = './log/'+lc_lm+'.log'
    if os.path.exists(logger_file):
        os.remove(logger_file)

    reload(logging)
    logging.basicConfig(filename=logger_file, \
                            level=logging.INFO)

    print("lambda_pitch - {}".format(lambda_pitch))
    print("lambda_mfc - {}".format(lambda_mfc))
    print("lambda_momenta - {}".format(lambda_momenta))

    logging.info("lambda_pitch - {}".format(lambda_pitch))
    logging.info("lambda_mfc - {}".format(lambda_mfc))
    logging.info("lambda_momenta - {}".format(lambda_momenta))

    if not os.path.isdir("./pitch_spect/"+lc_lm):
        os.makedirs(os.path.join("./pitch_spect/", lc_lm))
    else:
        for f in glob(os.path.join("./pitch_spect/", \
                lc_lm, "*.png")):
            os.remove(f)
    
    print('Preprocessing Data...')

    start_time = time.time()

    data_train = scio.loadmat(os.path.join(train_dir, 'cmu-arctic.mat'))

    pitch_A_train = np.expand_dims(data_train['src_f0_feat'], axis=-1)
    pitch_B_train = np.expand_dims(data_train['tar_f0_feat'], axis=-1)
    mfc_A_train = data_train['src_mfc_feat']
    mfc_B_train = data_train['tar_mfc_feat']
    momenta_A2B_train = np.expand_dims(data_train['momenta_f0_A2B'], axis=-1)
    momenta_B2A_train = np.expand_dims(data_train['momenta_f0_B2A'], axis=-1)

    end_time = time.time()
    time_elapsed = end_time - start_time

    print('Preprocessing Done.')

    print('Time Elapsed for Data Preprocessing: %02d:%02d:%02d' % (time_elapsed // 3600, \
                                                                   (time_elapsed % 3600 // 60), \
                                                                   (time_elapsed % 60 // 1)))
    

    # use pre_train arg to provide trained model
    model = VariationalCycleGAN(dim_pitch=23, dim_mfc=1, \
                n_frames=n_frames, pre_train=pre_train)
    
    for epoch in range(1,num_epochs+1):

        print('Epoch: %d' % epoch)
        logging.info('Epoch: %d' % epoch)

        start_time_epoch = time.time()

        mfc_A, pitch_A, momenta_A2B, mfc_B, pitch_B, momenta_B2A \
                = preproc.sample_data_momenta(mfc_A=mfc_A_train, mfc_B=mfc_B_train, 
                        pitch_A=pitch_A_train, pitch_B=pitch_B_train, 
                        momenta_A2B=momenta_A2B_train, 
                        momenta_B2A=momenta_B2A_train)

        print(mfc_A.shape, mfc_B.shape)
        sys.stdout.flush()
        
        n_samples = mfc_A.shape[0]
        
        train_mom_A2B_loss = list()
        train_mom_B2A_loss = list()

        train_pitch_A2B_loss = list()
        train_pitch_B2A_loss = list()

        train_mfc_A2B_loss = list()
        train_mfc_B2A_loss = list()

        for i in range(n_samples // mini_batch_size):

            start = i * mini_batch_size
            end = (i + 1) * mini_batch_size

            mom_loss_A2B, mom_loss_B2A, pitch_loss_A2B, \
            pitch_loss_B2A, mfc_loss_A2B, mfc_loss_B2A, \
            gen_mom_A, gen_pitch_A, gen_mfc_A, gen_mom_B, \
            gen_pitch_B, gen_mfc_B = model.train(mfc_A=mfc_A[start:end], 
                    mfc_B=mfc_B[start:end], pitch_A=pitch_A[start:end], 
                    pitch_B=pitch_B[start:end], momenta_A2B=momenta_A2B[start:end], 
                    momenta_B2A=momenta_B2A[start:end], lambda_pitch=lambda_pitch, 
                    lambda_mfc=lambda_mfc, lambda_momenta=lambda_momenta, 
                    generator_learning_rate=generator_learning_rate)
            
            train_mom_A2B_loss.append(mom_loss_A2B)
            train_mom_B2A_loss.append(mom_loss_B2A)

            train_pitch_A2B_loss.append(pitch_loss_A2B)
            train_pitch_B2A_loss.append(pitch_loss_B2A)

            train_mfc_A2B_loss.append(mfc_loss_A2B)
            train_mfc_B2A_loss.append(mfc_loss_B2A)
        
        print("Train Momenta A2B Loss- {}".format(np.mean(train_mom_A2B_loss)))
        print("Train Momenta B2A Loss- {}".format(np.mean(train_mom_B2A_loss)))
        
        logging.info("Train Momenta A2B Loss- {}".format(np.mean(train_mom_A2B_loss)))
        logging.info("Train Momenta B2A Loss- {}".format(np.mean(train_mom_B2A_loss)))

        logging.info("Train Pitch A2B Loss- {}".format(np.mean(train_pitch_A2B_loss)))
        logging.info("Train Pitch B2A Loss- {}".format(np.mean(train_pitch_B2A_loss)))

        logging.info("Train Mfc A2B Loss- {}".format(np.mean(train_mfc_A2B_loss)))
        logging.info("Train Mfc B2A Loss- {}".format(np.mean(train_mfc_B2A_loss)))

        end_time_epoch = time.time()
        time_elapsed_epoch = end_time_epoch - start_time_epoch

        print('Time Elapsed for This Epoch: %02d:%02d:%02d' % (time_elapsed_epoch // 3600, \
                (time_elapsed_epoch % 3600 // 60), (time_elapsed_epoch % 60 // 1)))

        logging.info('Time Elapsed for This Epoch: %02d:%02d:%02d' % (time_elapsed_epoch // 3600, \
                (time_elapsed_epoch % 3600 // 60), (time_elapsed_epoch % 60 // 1))) 

        if epoch%50==0:
            model.save(model_dir, model_name+str(epoch)+'.ckpt')


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

    parser.add_argument('--train_dir', type=str, help='Directory for A.', \
                        default=train_dir_default)
    parser.add_argument('--model_dir', type=str, help='Directory for saving models.', \
                        default=model_dir_default)
    parser.add_argument('--model_name', type=str, help='File name for saving model.', \
                        default=model_name_default)
    parser.add_argument('--random_seed', type=int, help='Random seed for model training.', \
                        default=random_seed_default)
    parser.add_argument('--tensorboard_log_dir', type=str, help='TensorBoard log directory.', \
                        default=tensorboard_log_dir_default)
    parser.add_argument('--current_iter', type = int, \
                        help = "Current iteration of the model (Fine tuning)", default=1)
    parser.add_argument("--lambda_pitch", type=float, help="hyperparam for cycle loss pitch", \
                        default=0.0001)#0.0001
    parser.add_argument("--lambda_mfc", type=float, help="hyperparam for cycle loss mfc", \
                        default=0.0001)
    parser.add_argument("--lambda_momenta", type=float, help="hyperparam for momenta magnitude", \
                        default=0.01)#0.1

    argv = parser.parse_args()

    train_dir = argv.train_dir
    model_dir = argv.model_dir
    model_name = argv.model_name
    random_seed = argv.random_seed
    tensorboard_log_dir = argv.tensorboard_log_dir

    lambda_pitch = argv.lambda_pitch
    lambda_mfc = argv.lambda_mfc
    lambda_momenta = argv.lambda_momenta

    train(train_dir=train_dir, model_dir=model_dir, model_name=model_name, 
          random_seed=random_seed, tensorboard_log_dir=tensorboard_log_dir, 
          pre_train=None, lambda_pitch=lambda_pitch, 
          lambda_mfc=lambda_mfc, lambda_momenta=lambda_momenta)
