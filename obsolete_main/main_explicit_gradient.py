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
from nn_models.model_explicit_gradient import VariationalCycleGAN
from utils.helper import smooth, generate_interpolation
import utils.preprocess as preproc


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def train(train_dir, model_dir, model_name, random_seed, \
            validation_dir, output_dir, \
            tensorboard_log_dir, pre_train=None, \
            lambda_cycle_pitch=0, lambda_cycle_mfc=0, lambda_momenta=0):

    np.random.seed(random_seed)

    num_epochs = 1000
    mini_batch_size = 1 # mini_batch_size = 1 is better

    generator_learning_rate = 0.0000001
    discriminator_learning_rate = 0.0000001

    sampling_rate = 16000
    num_mcep = 23
    frame_period = 5
    n_frames = 128

    lc_lm = "lp_"+str(lambda_cycle_pitch) \
            + '_lm_'+str(lambda_cycle_mfc) \
            +"_lmo_"+str(lambda_momenta) + '_msg'

    model_dir = os.path.join(model_dir, lc_lm)

    logger_file = './log/'+lc_lm+'.log'
    if os.path.exists(logger_file):
        os.remove(logger_file)

    logging.basicConfig(filename=logger_file, \
                            level=logging.DEBUG)

    print("lambda_cycle pitch - {}".format(lambda_cycle_pitch))
    print("lambda_cycle mfc - {}".format(lambda_cycle_mfc))
    print("lambda_momenta - {}".format(lambda_momenta))
    print("cycle_loss - L1")

    logging.info("lambda_cycle_pitch - {}".format(lambda_cycle_pitch))
    logging.info("lambda_cycle_mfc - {}".format(lambda_cycle_mfc))
    logging.info("lambda_momenta - {}".format(lambda_momenta))

    if not os.path.isdir("./pitch_spect/"+lc_lm):
        os.makedirs(os.path.join("./pitch_spect/", lc_lm))
    else:
        for f in glob(os.path.join("./pitch_spect/", \
                lc_lm, "*.png")):
            os.remove(f)
    
    print('Preprocessing Data...')

    start_time = time.time()

    data_train = scio.loadmat(os.path.join(train_dir, 'train_5.mat'))
    data_valid = scio.loadmat(os.path.join(train_dir, 'valid_5.mat'))

    pitch_A_train = np.expand_dims(data_train['src_f0_feat'], axis=-1)
    pitch_B_train = np.expand_dims(data_train['tar_f0_feat'], axis=-1)
    mfc_A_train = data_train['src_mfc_feat']
    mfc_B_train = data_train['tar_mfc_feat']
    
    pitch_A_valid = np.expand_dims(data_valid['src_f0_feat'], axis=-1)
    pitch_B_valid = np.expand_dims(data_valid['tar_f0_feat'], axis=-1)
    mfc_A_valid = data_valid['src_mfc_feat']
    mfc_B_valid = data_valid['tar_mfc_feat']

    # Randomly shuffle the trainig data
    indices_train = np.arange(0, pitch_A_train.shape[0])
    np.random.shuffle(indices_train)
    pitch_A_train = pitch_A_train[indices_train]
    mfc_A_train = mfc_A_train[indices_train]
    np.random.shuffle(indices_train)
    pitch_B_train = pitch_B_train[indices_train]
    mfc_B_train = mfc_B_train[indices_train]

    mfc_A_valid, pitch_A_valid, \
        mfc_B_valid, pitch_B_valid = preproc.sample_data(mfc_A=mfc_A_valid, \
                                    mfc_B=mfc_B_valid, pitch_A=pitch_A_valid, \
                                    pitch_B=pitch_B_valid)

    if validation_dir is not None:
        validation_output_dir = os.path.join(output_dir, lc_lm)
        if not os.path.exists(validation_output_dir):
            os.makedirs(validation_output_dir)

    end_time = time.time()
    time_elapsed = end_time - start_time

    print('Preprocessing Done.')

    print('Time Elapsed for Data Preprocessing: %02d:%02d:%02d' % (time_elapsed // 3600, \
                                                                   (time_elapsed % 3600 // 60), \
                                                                   (time_elapsed % 60 // 1)))
    
    #use pre_train arg to provide trained model
    model = VariationalCycleGAN(dim_pitch=1, dim_mfc=23, \
                n_frames=n_frames, pre_train=pre_train)
    
    for epoch in range(1,num_epochs+1):

        print('Epoch: %d' % epoch)
        logging.info('Epoch: %d' % epoch)

        start_time_epoch = time.time()

        mfc_A, pitch_A, \
            mfc_B, pitch_B = preproc.sample_data(mfc_A=mfc_A_train, \
                            mfc_B=mfc_B_train, pitch_A=pitch_A_train, \
                            pitch_B=pitch_B_train)
        
        n_samples = mfc_A.shape[0]
        
        train_gen_loss = []
        train_disc_loss = []

        for i in range(n_samples // mini_batch_size):

            start = i * mini_batch_size
            end = (i + 1) * mini_batch_size

            generator_loss, discriminator_loss, \
            gen_pitch_A, gen_mfc_A, gen_pitch_B, \
            gen_mfc_B, mom_A, mom_B, gen_grad, disc_grad \
                = model.train_grad(mfc_A=mfc_A[start:end], 
                    mfc_B=mfc_B[start:end], pitch_A=pitch_A[start:end], 
                    pitch_B=pitch_B[start:end], lambda_cycle_pitch=lambda_cycle_pitch, 
                    lambda_cycle_mfc=lambda_cycle_mfc, lambda_momenta=lambda_momenta, 
                    generator_learning_rate=generator_learning_rate, 
                    discriminator_learning_rate=discriminator_learning_rate)
            
            if (i+1)%50 == 0:

                pylab.figure(figsize=(13,13))
                pylab.subplot(221)
                pylab.hist(np.reshape(np.divide(gen_grad[0][0], 1e-10+gen_grad[0][1]), 
                                      (-1,)), bins=100, facecolor='red', alpha=0.5, 
                    label='Sampler 1')
                pylab.legend(loc=2)
                pylab.subplot(222)
                pylab.hist(np.reshape(np.divide(gen_grad[62][0], 1e-10+gen_grad[62][1]), 
                                      (-1,)), bins=100, facecolor='blue', alpha=0.5, 
                    label='Generator 1')
                pylab.legend(loc=2)

                pylab.subplot(223)
                pylab.hist(np.reshape(np.divide(gen_grad[136][0], 1e-10+gen_grad[136][1]), 
                                      (-1,)), bins=100, facecolor='red', alpha=0.5, 
                    label='Sampler 2')
                pylab.legend(loc=2)
                pylab.subplot(224)
                pylab.hist(np.reshape(np.divide(gen_grad[198][0], 1e-10+gen_grad[198][1]), 
                                      (-1,)), bins=100, facecolor='blue', alpha=0.5, 
                    label='Generator 2')
                pylab.legend(loc=2)

                pylab.suptitle('Epoch '+str(epoch)+' example '+str(i+1))
                pylab.savefig('./pitch_spect/'+lc_lm+'/'\
                        +'grads_'+str(epoch)+'_'+str(i+1)+'.png')
                pylab.close()
            
            train_gen_loss.append(generator_loss)
            train_disc_loss.append(discriminator_loss)

        
        print("Train Generator Loss- {}".format(np.mean(train_gen_loss)))
        print("Train Discriminator Loss- {}".format(np.mean(train_disc_loss)))
        
        logging.info("Train Generator Loss- {}".format(np.mean(train_gen_loss)))
        logging.info("Train Discriminator Loss- {}".format(np.mean(train_disc_loss)))
        
        end_time_epoch = time.time()
        time_elapsed_epoch = end_time_epoch - start_time_epoch

        print('Time Elapsed for This Epoch: %02d:%02d:%02d' % (time_elapsed_epoch // 3600, \
                (time_elapsed_epoch % 3600 // 60), (time_elapsed_epoch % 60 // 1)))

        logging.info('Time Elapsed for This Epoch: %02d:%02d:%02d' % (time_elapsed_epoch // 3600, \
                (time_elapsed_epoch % 3600 // 60), (time_elapsed_epoch % 60 // 1))) 


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description = 'Train VariationalCycleGAN model for datasets.')

    emo_dict = {"neu-ang":['neutral', 'angry'], \
                        "neu-sad":['neutral', 'sad'], \
                        "neu-hap":['neutral', 'happy']}

    emo_pair = "neu-ang"
    train_dir_default = "./data/"+emo_pair
    model_dir_default = "./model/"+emo_pair
    model_name_default = emo_pair
    validation_dir_default = './data/evaluation/'+emo_pair+"/"+emo_dict[emo_pair][0]+'_5'
#    validation_dir_default = './data/evaluation/'+emo_pair+"/"+emo_dict[emo_pair][0]
    output_dir_default = './validation_output/'+emo_pair
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
    parser.add_argument('--validation_dir', type=str, \
                        help='Convert validation after each training epoch. Set None for no conversion', \
                        default=validation_dir_default)
    parser.add_argument('--output_dir', type=str, \
                        help='Output directory for converted validation voices.', default=output_dir_default)
    parser.add_argument('--tensorboard_log_dir', type=str, help='TensorBoard log directory.', \
                        default=tensorboard_log_dir_default)
    parser.add_argument('--current_iter', type = int, \
                        help = "Current iteration of the model (Fine tuning)", default=1)
    parser.add_argument("--lambda_cycle_pitch", type=float, help="hyperparam for cycle loss pitch", \
                        default=0.00001)
    parser.add_argument("--lambda_cycle_mfc", type=float, help="hyperparam for cycle loss mfc", \
                        default=0.1)
    parser.add_argument("--lambda_momenta", type=float, help="hyperparam for momenta magnitude", \
                        default=1e-4)

    argv = parser.parse_args()

    train_dir = argv.train_dir
    model_dir = argv.model_dir
    model_name = argv.model_name
    random_seed = argv.random_seed
    validation_dir = None if argv.validation_dir == 'None' or argv.validation_dir == 'none' \
                        else argv.validation_dir
    output_dir = argv.output_dir
    tensorboard_log_dir = argv.tensorboard_log_dir

    lambda_cycle_pitch = argv.lambda_cycle_pitch
    lambda_cycle_mfc = argv.lambda_cycle_mfc
    lambda_momenta = argv.lambda_momenta

    train(train_dir=train_dir, model_dir=model_dir, model_name=model_name, 
          random_seed=random_seed, validation_dir=validation_dir, 
          output_dir=output_dir, tensorboard_log_dir=tensorboard_log_dir, 
          pre_train=None, lambda_cycle_pitch=lambda_cycle_pitch, 
          lambda_cycle_mfc=lambda_cycle_mfc, lambda_momenta=lambda_momenta)
