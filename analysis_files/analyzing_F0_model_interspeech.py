import argparse
import os
import numpy as np
import librosa
import scipy.io.wavfile as scwav
import scipy
import scipy.signal as scisig
import scipy.io as scio
import pylab
import tensorflow as tf

import utils.preprocess as preproc
from utils.helper import smooth, generate_interpolation
from utils.model_utils import delta_matrix
from model_pair_lvi import CycleGAN as CycleGAN_f0
from mfcc_spect_analysis_VCGAN import _power_to_db
from scipy.linalg import sqrtm, inv


num_mfcc = 23
num_pitch = 1
sampling_rate = 16000
frame_period = 5.0


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


def sym(w):
    return w.dot(inv(sqrtm(w.T.dot(w))))


def normalize(x, nmz_type='min_max'):
    """
    x is the data to be normalized MxN matrix
    nmz_type - [min_max, max, mean_var]
    """
    if nmz_type == 'min_max':
        x = (x - np.min(x)) / (np.max(x) - np.min(x))
    elif nmz_type == 'max':
        x = x / (np.max(x) + 1e-20)
    elif nmz_type == 'mean_var':
        x = (x - np.mean(x)) / np.std(x)
    else:
        raise Exception('normalization type not recognized')
    return x


if __name__ == '__main__':
    data_valid = scio.loadmat('/home/ravi/Desktop/spect-pitch-gan/data/neu-ang/valid_mod_dtw_harvest.mat')
    
    pitch_A_valid = np.expand_dims(data_valid['src_f0_feat'], axis=-1)
    pitch_B_valid = np.expand_dims(data_valid['tar_f0_feat'], axis=-1)
    pitch_A_valid = np.transpose(pitch_A_valid, (0,1,3,2))
    pitch_B_valid = np.transpose(pitch_B_valid, (0,1,3,2))
    mfc_A_valid = np.transpose(data_valid['src_mfc_feat'], (0,1,3,2))
    mfc_B_valid = np.transpose(data_valid['tar_mfc_feat'], (0,1,3,2))
    
#    mfc_A_valid, pitch_A_valid, \
#        mfc_B_valid, pitch_B_valid = preproc.sample_data(mfc_A=mfc_A_valid, \
#                                    mfc_B=mfc_B_valid, pitch_A=pitch_A_valid, \
#                                    pitch_B=pitch_B_valid)

    mfc_A_valid = np.vstack(mfc_A_valid)
    mfc_B_valid = np.vstack(mfc_B_valid)
    pitch_A_valid = np.vstack(pitch_A_valid)
    pitch_B_valid = np.vstack(pitch_B_valid)
    
    cgan_f0 = CycleGAN_f0(mode='test')
    cgan_f0.load(filepath='/home/ravi/Desktop/pitch-gan/pitch-lddmm-gan/model_f0/neu-ang/selected/neu-ang.ckpt')
    
    f0_conv = np.empty((0,128))
    f0_valid = np.empty((0,128))
    f0_input = np.empty((0,128))

    for i in range(mfc_A_valid.shape[0]):

        pred_f0 = cgan_f0.test(input_pitch=pitch_A_valid[i:i+1], 
                               input_mfc=mfc_A_valid[i:i+1], 
                               direction='A2B')
        
        f0_conv = np.concatenate((f0_conv, pred_f0.reshape(1,-1)), axis=0)   
        f0_valid = np.concatenate((f0_valid, pitch_B_valid[i:i+1].reshape(1,-1)), axis=0)
        f0_input = np.concatenate((f0_input, pitch_A_valid[i:i+1].reshape(1,-1)), axis=0)
    
    del pred_f0
    






















