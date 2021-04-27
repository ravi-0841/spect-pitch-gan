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
from nn_models.model_energy_f0_momenta_wasserstein import VariationalCycleGAN as VCGAN
#from mfcc_spect_analysis_VCGAN import _power_to_db
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
    data_valid = scio.loadmat('/home/ravi/Desktop/spect-pitch-gan/data/neu-ang/neu-ang_unaligned_valid_sum_mfc.mat')
    
    pitch_A_valid = data_valid['src_f0_feat']
    pitch_B_valid = data_valid['tar_f0_feat']
    energy_A_valid = data_valid['src_ec_feat']
    energy_B_valid = data_valid['tar_ec_feat']
    
    pitch_A_valid = np.transpose(pitch_A_valid, (0,1,3,2))
    pitch_B_valid = np.transpose(pitch_B_valid, (0,1,3,2))
    energy_A_valid = np.transpose(energy_A_valid, (0,1,3,2))
    energy_B_valid = np.transpose(energy_B_valid, (0,1,3,2))
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
    energy_A_valid = np.vstack(energy_A_valid)
    energy_B_valid = np.vstack(energy_B_valid)

    model = VCGAN(dim_mfc=23, dim_pitch=1, dim_energy=1, mode='test')
    model.load(filepath='/home/ravi/Desktop/F0_sum_ec/mixed_and_raw_models/sum_mfc_models/neu-ang/lp_1e-05_le_0.1_li_0.0_lrg_1e-05_lrd_1e-07_sum_mfc_epoch_200_best/neu-ang_200.ckpt')

    f0_conv = np.empty((0,128))
    f0_valid = np.empty((0,128))
    f0_input = np.empty((0,128))
    cyc_f0 = np.empty((0,128))

    ec_conv = np.empty((0,128))
    ec_valid = np.empty((0,128))
    ec_input = np.empty((0,128))
    cyc_ec = np.empty((0,128))

    for i in range(mfc_A_valid.shape[0]):

        f0_converted, _, ec_converted, _ = model.test(input_pitch=pitch_A_valid[i:i+1], 
                                                      input_mfc=mfc_A_valid[i:i+1],
                                                      input_energy=energy_A_valid[i:i+1],
                                                      direction='A2B')
        
        mfc_converted = np.multiply(mfc_A_valid[i:i+1], np.divide(ec_converted, energy_A_valid[i:i+1]))
        cyc_pred_f0, _, cyc_pred_ec, _ = model.test(input_pitch=f0_converted, 
                                                    input_mfc=mfc_converted,
                                                    input_energy=ec_converted,
                                                    direction='B2A')
        
        f0_conv = np.concatenate((f0_conv, f0_converted.reshape(1,-1)), axis=0)
        cyc_f0 = np.concatenate((cyc_f0, cyc_pred_f0.reshape(1,-1)), axis=0)
        
        ec_conv = np.concatenate((ec_conv, ec_converted.reshape(1,-1)), axis=0)
        cyc_ec = np.concatenate((cyc_ec, cyc_pred_ec.reshape(1,-1)), axis=0)
        
        f0_valid = np.concatenate((f0_valid, pitch_B_valid[i:i+1].reshape(1,-1)), axis=0)
        f0_input = np.concatenate((f0_input, pitch_A_valid[i:i+1].reshape(1,-1)), axis=0)
        
        ec_valid = np.concatenate((ec_valid, energy_B_valid[i:i+1].reshape(1,-1)), axis=0)
        ec_input = np.concatenate((ec_input, energy_A_valid[i:i+1].reshape(1,-1)), axis=0)

    del f0_converted, mfc_converted, ec_converted, cyc_pred_f0, cyc_pred_ec






















