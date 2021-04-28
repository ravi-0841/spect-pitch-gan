import argparse
import os
import numpy as np
import tensorflow as tf
import scipy.io as scio

from model_f0 import CycleGAN as CycleGAN_f0
from model_mceps import CycleGAN as CycleGAN_mceps

from preprocess import *
from utils import get_lf0_cwt_norm,norm_scale,denormalize
from utils import get_cont_lf0, get_lf0_cwt,inverse_cwt
from sklearn import preprocessing

os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3"

tf.reset_default_graph()

def f0_conversion(model_f0_dir, model_f0_name, features, direction):
    tf.reset_default_graph()
    model_f0 = CycleGAN_f0(num_features = 10, mode = 'test')
    model_f0.load(filepath=os.path.join(model_f0_dir,model_f0_name))
    lf0 = model_f0.test(inputs=features, direction=conversion_direction)[0]
    return lf0

def mcep_conversion(model_mceps_dir, model_mceps_name, features, direction):
    tf.reset_default_graph()
    model_mceps = CycleGAN_mceps(num_features = 24, mode = 'test')
    model_mceps.load(filepath=os.path.join(model_mceps_dir,model_mceps_name))
    coded_sp_converted_norm = model_mceps.test(inputs=features, \
                    direction=conversion_direction)[0]
    return coded_sp_converted_norm


def conversion(model_f0_dir, model_f0_name, model_mceps_dir, model_mceps_name, \
        data_dir, conversion_direction, output_dir):

    num_mceps = 24
    sampling_rate = 16000
    frame_period = 5.0

    mcep_normalization_params = np.load(os.path.join(model_mceps_dir, 'mcep_normalization.npz'))
    mcep_mean_A = mcep_normalization_params['mean_A']
    mcep_std_A = mcep_normalization_params['std_A']
    mcep_mean_B = mcep_normalization_params['mean_B']
    mcep_std_B = mcep_normalization_params['std_B']

    logf0s_normalization_params = np.load(os.path.join(model_f0_dir, 'logf0s_normalization.npz'))
    logf0s_mean_A = logf0s_normalization_params['mean_A']
    logf0s_std_A = logf0s_normalization_params['std_A']
    logf0s_mean_B = logf0s_normalization_params['mean_B']
    logf0s_std_B = logf0s_normalization_params['std_B']
    
    conv_f0s = np.empty((0,128))

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    data_valid = scio.loadmat('/home/ravi/Desktop/spect-pitch-gan/data/neu-sad/neu-sad_unaligned_valid_sum_mfc.mat')
    
    pitch_A_valid = data_valid['src_f0_feat']
    pitch_B_valid = data_valid['tar_f0_feat']
    pitch_A_valid = np.transpose(pitch_A_valid, (0,1,3,2))
    pitch_B_valid = np.transpose(pitch_B_valid, (0,1,3,2))
    pitch_A_valid = np.vstack(pitch_A_valid)
    pitch_B_valid = np.vstack(pitch_B_valid)

    model_f0 = CycleGAN_f0(num_features=10, mode='test')
    model_f0.load(filepath=os.path.join(model_f0_dir, model_f0_name))

    for pitch in pitch_A_valid:
        
        try:
            f0 = pitch.reshape((-1,))
            
            uv, cont_lf0_lpf = get_cont_lf0(f0)
    
            if conversion_direction == 'A2B':
    
                cont_lf0_lpf_norm = (cont_lf0_lpf - logf0s_mean_A) / logf0s_std_A
                Wavelet_lf0, scales = get_lf0_cwt(cont_lf0_lpf_norm)
                Wavelet_lf0_norm, mean, std = norm_scale(Wavelet_lf0)
                lf0_cwt_norm = Wavelet_lf0_norm.T
                
                lf0 = model_f0.test(inputs=np.array([lf0_cwt_norm]), 
                                    direction=conversion_direction)[0]
    
                lf0_cwt_denormalize = denormalize(lf0.T, mean, std)
                lf0_rec = inverse_cwt(lf0_cwt_denormalize,scales)
                lf0_converted = lf0_rec * logf0s_std_B + logf0s_mean_B
                f0_converted = np.squeeze(uv) * np.exp(lf0_converted)
                f0_converted = np.ascontiguousarray(f0_converted)
                
                conv_f0s = np.concatenate((conv_f0s, f0_converted.reshape(1,-1)))

            print("Processed")
        except Exception as ex:
            print(ex)
    
    return conv_f0s


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Convert voices using pre-trained CycleGAN model.')
    
    emo_pair_default = 'neutral_to_sad'
    emo_pair_dict = {'neutral_to_angry':'neu-ang', 'neutral_to_happy':'neu-hap', \
                     'neutral_to_sad':'neu-sad'}
    
    parser.add_argument('--emo_pair', type=str, help='Emotion pair', \
                        default=emo_pair_default, \
                        choices=['neutral_to_angry', 'neutral_to_happy', \
                                 'neutral_to_sad'])

    argv = parser.parse_args()
    
    emo_pair = argv.emo_pair
    target = emo_pair.split('_')[-1]
    
    model_f0_dir = './model/'+emo_pair+'_f0'
    model_f0_name = emo_pair+'_f0.ckpt'
    model_mceps_dir = './model/'+emo_pair+'_oos_mceps'
    model_mceps_name = emo_pair+'_mceps.ckpt'
    data_dir = '/home/ravi/Downloads/Emo-Conv/neutral-{0}/valid/neutral'.format(target)
#    data_dir = '../data/evaluation/'+emo_pair_dict[emo_pair]+'/test_oos/neutral'
#    data_dir = '/home/ravi/Desktop/Pitch-Energy/Wavenet-tts-samples/speech_US/fine-tune-'\
#                +target+'/test'
    conversion_direction = 'A2B'
    output_dir = './converted_test/'+emo_pair+'/oos/'

    conv_f0s = conversion(model_f0_dir = model_f0_dir, model_f0_name = model_f0_name, \
            model_mceps_dir = model_mceps_dir, model_mceps_name = model_mceps_name, \
            data_dir = data_dir, conversion_direction = conversion_direction, \
            output_dir = output_dir)


