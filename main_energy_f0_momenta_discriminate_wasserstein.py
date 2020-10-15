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
from nn_models.model_energy_f0_momenta_discriminate_wasserstein import VariationalCycleGAN
from utils.helper import smooth, generate_interpolation
import utils.preprocess as preproc
from importlib import reload


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def train(train_dir, model_dir, model_name, random_seed, \
            validation_dir, output_dir, \
            tensorboard_log_dir, pre_train=None, \
            lambda_cycle_pitch=0, lambda_cycle_energy=0, lambda_momenta=0, 
            lambda_identity_energy=0, generator_learning_rate=1e-05, 
            discriminator_learning_rate=1e-03, emo_pair='neu-ang'):

    np.random.seed(random_seed)

    num_epochs = 1000
    mini_batch_size = 1 # mini_batch_size = 1 is better

    sampling_rate = 16000
    num_mcep = 23
    frame_period = 5
    n_frames = 128

    lc_lm = 'lp_'+str(lambda_cycle_pitch) \
            + '_le_'+str(lambda_cycle_energy) \
            + '_li_'+str(lambda_identity_energy) \
            +'_lrg_'+str(generator_learning_rate) \
            +'_lrd_'+str(discriminator_learning_rate) \
            + '_pre_trained'

    model_dir = os.path.join(model_dir, lc_lm)

    logger_file = './log/'+lc_lm+'.log'
    if os.path.exists(logger_file):
        os.remove(logger_file)

    reload(logging)
    logging.basicConfig(filename=logger_file, \
                            level=logging.DEBUG)

    print("lambda_cycle pitch - {}".format(lambda_cycle_pitch))
    print("lambda_cycle energy - {}".format(lambda_cycle_energy))
    print("lambda_momenta - {}".format(lambda_momenta))

    logging.info("lambda_cycle_pitch - {}".format(lambda_cycle_pitch))
    logging.info("lambda_cycle_energy - {}".format(lambda_cycle_energy))
    logging.info("lambda_identity_energy - {}".format(lambda_identity_energy))
    logging.info("lambda_momenta - {}".format(lambda_momenta))
    logging.info("generator_lr - {}".format(generator_learning_rate))
    logging.info("discriminator_lr - {}".format(discriminator_learning_rate))

    if not os.path.isdir("./pitch_spect/"+lc_lm):
        os.makedirs(os.path.join("./pitch_spect/", lc_lm))
    else:
        for f in glob(os.path.join("./pitch_spect/", lc_lm, "*.png")):
            os.remove(f)
    
    print('Preprocessing Data...')

    start_time = time.time()

    data_train = scio.loadmat(os.path.join(train_dir, 'unaligned_train_no_ec_process_5.mat'))
    data_valid = scio.loadmat(os.path.join(train_dir, 'unaligned_valid_no_ec_process_5.mat'))

    pitch_A_train = data_train['src_f0_feat']
    pitch_B_train = data_train['tar_f0_feat']
    energy_A_train = np.log(data_train['src_ec_feat'])
    energy_B_train = np.log(data_train['tar_ec_feat'])
    mfc_A_train = data_train['src_mfc_feat']
    mfc_B_train = data_train['tar_mfc_feat']

    pitch_A_valid = data_valid['src_f0_feat']
    pitch_B_valid = data_valid['tar_f0_feat']
    energy_A_valid = np.log(data_valid['src_ec_feat'])
    energy_B_valid = np.log(data_valid['tar_ec_feat'])
    mfc_A_valid = data_valid['src_mfc_feat']
    mfc_B_valid = data_valid['tar_mfc_feat']

    # Randomly shuffle the trainig data
    indices_train = np.arange(0, pitch_A_train.shape[0])
    np.random.shuffle(indices_train)
    pitch_A_train = pitch_A_train[indices_train]
    energy_A_train = energy_A_train[indices_train]
    mfc_A_train = mfc_A_train[indices_train]

    np.random.shuffle(indices_train)
    pitch_B_train = pitch_B_train[indices_train]
    energy_B_train = energy_B_train[indices_train]
    mfc_B_train = mfc_B_train[indices_train]

    mfc_A_valid, pitch_A_valid, energy_A_valid, \
        mfc_B_valid, pitch_B_valid, energy_B_valid = preproc.sample_data_energy(mfc_A=mfc_A_valid, 
                mfc_B=mfc_B_valid, pitch_A=pitch_A_valid, pitch_B=pitch_B_valid, 
                energy_A=energy_A_valid, energy_B=energy_B_valid)

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
    model = VariationalCycleGAN(dim_pitch=1, dim_energy=1, dim_mfc=23, n_frames=n_frames, 
                                pre_train=pre_train, log_file_name=lc_lm)
    
    for epoch in range(1,num_epochs+1):

        print('Epoch: %d' % epoch)
        logging.info('Epoch: %d' % epoch)

        start_time_epoch = time.time()

        mfc_A, pitch_A, energy_A, \
            mfc_B, pitch_B, energy_B = preproc.sample_data_energy(mfc_A=mfc_A_train, 
                    mfc_B=mfc_B_train, pitch_A=pitch_A_train, pitch_B=pitch_B_train, 
                    energy_A=energy_A_train, energy_B=energy_B_train)
        
        n_samples = energy_A.shape[0]
        
        train_gen_loss = []
        train_disc_loss = []

        for i in range(n_samples // mini_batch_size):

            start = i * mini_batch_size
            end = (i + 1) * mini_batch_size

            generator_loss, discriminator_loss, \
            gen_pitch_A, gen_energy_A, gen_pitch_B, \
            gen_energy_B, mom_pitch_A, mom_pitch_B, \
            mom_energy_A, mom_energy_B \
                = model.train(mfc_A=mfc_A[start:end], energy_A=energy_A[start:end], 
                    pitch_A=pitch_A[start:end], mfc_B=mfc_B[start:end],  
                    energy_B=energy_B[start:end], pitch_B=pitch_B[start:end], 
                    lambda_cycle_pitch=lambda_cycle_pitch, 
                    lambda_cycle_energy=lambda_cycle_energy, 
                    lambda_momenta=lambda_momenta,
                    lambda_identity_energy=lambda_identity_energy, 
                    generator_learning_rate=generator_learning_rate, 
                    discriminator_learning_rate=discriminator_learning_rate)
            
            train_gen_loss.append(generator_loss)
            train_disc_loss.append(discriminator_loss)

        
        print("Train Generator Loss- {}".format(np.mean(train_gen_loss)))
        print("Train Discriminator Loss- {}".format(np.mean(train_disc_loss)))
        
        logging.info("Train Generator Loss- {}".format(np.mean(train_gen_loss)))
        logging.info("Train Discriminator Loss- {}".format(np.mean(train_disc_loss)))

        if epoch%100 == 0:

            for i in range(energy_A_valid.shape[0]):

                gen_pitch_A, gen_energy_A, \
                gen_pitch_B, gen_energy_B, \
                mom_pitch_A, mom_pitch_B, \
                mom_energy_A, mom_energy_B = model.test_gen(mfc_A=mfc_A_valid[i:i+1], 
                                mfc_B=mfc_B_valid[i:i+1], energy_A=energy_A_valid[i:i+1], 
                                energy_B=energy_B_valid[i:i+1], pitch_A=pitch_A_valid[i:i+1], 
                                pitch_B=pitch_B_valid[i:i+1])

                pylab.figure(figsize=(13,13))
                pylab.subplot(221)
                pylab.plot(pitch_A_valid[i].reshape(-1,), label='F0 A')
                pylab.plot(gen_pitch_B.reshape(-1,), label='F0 A2B')
                pylab.plot(pitch_B_valid[i].reshape(-1,), label='F0 B')
                pylab.plot(mom_pitch_B.reshape(-1,), label='momenta')
                pylab.legend(loc=2)
                pylab.subplot(222)
                pylab.plot(energy_A_valid[i].reshape(-1,), label='Energy A')
                pylab.plot(gen_energy_B.reshape(-1,), label='Energy A2B')
                pylab.plot(energy_B_valid[i].reshape(-1,), label='Energy B')
                pylab.plot(mom_energy_B.reshape(-1,), label='momenta')
                pylab.legend(loc=2)

                pylab.subplot(223)
                pylab.plot(pitch_B_valid[i].reshape(-1,), label='F0 B')
                pylab.plot(gen_pitch_A.reshape(-1,), label='F0 B2A')
                pylab.plot(pitch_A_valid[i].reshape(-1,), label='F0 A')
                pylab.plot(mom_pitch_A.reshape(-1,), label='momenta')
                pylab.legend(loc=2)
                pylab.subplot(224)
                pylab.plot(energy_B_valid[i].reshape(-1,), label='Energy B')
                pylab.plot(gen_energy_A.reshape(-1,), label='Energy B2A')
                pylab.plot(energy_A_valid[i].reshape(-1,), label='Energy A')
                pylab.plot(mom_energy_A.reshape(-1,), label='momenta')
                pylab.legend(loc=2)

                pylab.suptitle('Epoch '+str(epoch)+' example '+str(i+1))
                pylab.savefig('./pitch_spect/'+lc_lm+'/'\
                        +str(epoch)+'_'+str(i+1)+'.png')
                pylab.close()
        
        end_time_epoch = time.time()
        time_elapsed_epoch = end_time_epoch - start_time_epoch

        print('Time Elapsed for This Epoch: %02d:%02d:%02d' % (time_elapsed_epoch // 3600, \
                (time_elapsed_epoch % 3600 // 60), (time_elapsed_epoch % 60 // 1)))

        logging.info('Time Elapsed for This Epoch: %02d:%02d:%02d' % (time_elapsed_epoch // 3600, \
                (time_elapsed_epoch % 3600 // 60), (time_elapsed_epoch % 60 // 1)))

        if epoch % 100 == 0:
            
            cur_model_name = model_name+"_"+str(epoch)+".ckpt"
            model.save(directory=model_dir, filename=cur_model_name)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description = 'Train VariationalCycleGAN model for datasets.')

    emo_dict = {"neu-ang":['neutral', 'angry'], \
                        "neu-sad":['neutral', 'sad'], \
                        "neu-hap":['neutral', 'happy']}

    emo_pair_default = "neu-ang"
    random_seed_default = 0

    parser.add_argument('--random_seed', type=int, help='Random seed for model training.', 
            default=random_seed_default)
    parser.add_argument('--current_iter', type=int, help="Current iteration of the model (Fine tuning)", 
            default=1)
    parser.add_argument('--lambda_cycle_pitch', type=float, help="hyperparam for cycle loss pitch", 
            default=0.00001)
    parser.add_argument('--lambda_cycle_energy', type=float, help="hyperparam for cycle loss energy", 
            default=0.1)
    parser.add_argument('--lambda_identity_energy', type=float, help="hyperparam for identity loss energy", 
            default=0.05)
    parser.add_argument('--lambda_momenta', type=float, help="hyperparam for momenta magnitude", 
            default=1e-06)
    parser.add_argument('--generator_learning_rate', type=float, help="generator learning rate", 
            default=1e-06)
    parser.add_argument('--discriminator_learning_rate', type=float, help="discriminator learning rate", 
            default=1e-07)
    parser.add_argument('--emotion_pair', type=str, help="Emotion Pair", 
            default=emo_pair_default)
    
    argv = parser.parse_args()

    emo_pair = argv.emotion_pair
    train_dir = "./data/"+emo_pair
    model_dir = "./model/"+emo_pair
    model_name = emo_pair
    validation_dir = './data/evaluation/'+emo_pair+"/"+emo_dict[emo_pair][0]+'_5'
#    validation_dir = './data/evaluation/'+emo_pair+"/"+emo_dict[emo_pair][0]
    output_dir = './validation_output/'+emo_pair
    tensorboard_log_dir = './log/'+emo_pair

    random_seed = argv.random_seed

    lambda_cycle_pitch = argv.lambda_cycle_pitch
    lambda_cycle_energy = argv.lambda_cycle_energy
    lambda_identity_energy = argv.lambda_identity_energy*0.5
    lambda_momenta = argv.lambda_momenta

    generator_learning_rate = argv.generator_learning_rate
    discriminator_learning_rate = argv.discriminator_learning_rate

    train(train_dir=train_dir, model_dir=model_dir, model_name=model_name, 
          random_seed=random_seed, validation_dir=validation_dir, 
          output_dir=output_dir, tensorboard_log_dir=tensorboard_log_dir, 
          pre_train='./model/cmu-arctic/le_10.0_supervised_mwd_spect_male_female/cmu-arctic_850.ckpt', 
          lambda_cycle_pitch=lambda_cycle_pitch, lambda_cycle_energy=lambda_cycle_energy, 
          lambda_momenta=lambda_momenta, lambda_identity_energy=lambda_identity_energy,  
          generator_learning_rate=generator_learning_rate, 
          discriminator_learning_rate=discriminator_learning_rate, 
          emo_pair=argv.emotion_pair)
