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
import pyworld as pw

import utils.preprocess as preproc
from utils.helper import smooth, generate_interpolation
from utils.model_utils import delta_matrix
from nn_models.model_pitch_mfc_discriminate_wasserstein import VariationalCycleGAN
#from nn_models.model_embedding_wasserstein import VariationalCycleGAN
from encoder_decoder import AE
#from nn_models.model_wasserstein import VariationalCycleGAN
from mfcc_spect_analysis_VCGAN import _power_to_db
from scipy.linalg import sqrtm, inv


num_mfcc = 1
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

#    pitch_A_valid = np.transpose(data_valid['src_f0_feat'], (0,1,3,2))
#    pitch_B_valid = np.transpose(data_valid['tar_f0_feat'], (0,1,3,2))
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
    
    ae_model = AE(dim_mfc=23)
    ae_model.load(filename='./model/AE_cmu_pre_trained_noise_std_1.ckpt')
    model = VariationalCycleGAN(dim_mfc=num_mfcc, dim_pitch=num_pitch, mode='test')
#    model.load(filepath='./model/neu-ang/lp_1e-05_lm_0.1_lmo_1e-06_lrg_1e-06_lrd_1e-07_li_0.05_pre_trained_pitch_mfc_discriminate_wasserstein/neu-ang_2000.ckpt')
    model.load(filepath='/home/ravi/Desktop/spect-pitch-gan/model/neu-ang/lp_1e-05_lm_0.1_lmo_1e-06_lrg_2e-06_lrd_1e-07_li_0.05_pre_trained_pitch_mfc_discriminate_wasserstein_all_spk/neu-ang_450.ckpt')
    
    mfc_A_valid = ae_model.get_embedding(mfc_features=mfc_A_valid)
    mfc_B_valid = ae_model.get_embedding(mfc_features=mfc_B_valid)
    
    f0_conv = np.empty((0,128))
    f0_valid = np.empty((0,128))
    f0_input = np.empty((0,128))
    cyc_f0 = np.empty((0,128))
    mfc_conv = np.empty((0,128))
    cyc_mfc = np.empty((0,128))

    for i in range(mfc_A_valid.shape[0]):

        pred_f0, pred_mfc = model.test(input_pitch=pitch_A_valid[i:i+1], 
                                           input_mfc=mfc_A_valid[i:i+1], 
                                           direction='A2B')
        cyc_pred_f0, cyc_pred_mfc = model.test(input_pitch=pred_f0, 
                                               input_mfc=pred_mfc, 
                                               direction='B2A')
        
        f0_conv = np.concatenate((f0_conv, pred_f0.reshape(1,-1)), axis=0)
        cyc_f0 = np.concatenate((cyc_f0, cyc_pred_f0.reshape(1,-1)), axis=0)
        
        mfc_conv = np.concatenate((mfc_conv, pred_mfc.reshape(1,-1)), axis=0)
        cyc_mfc = np.concatenate((cyc_mfc, cyc_pred_mfc.reshape(1,-1)), axis=0)

        mfc_source = np.transpose(np.squeeze(mfc_A_valid[i]))        
        mfc_target = np.transpose(np.squeeze(mfc_B_valid[i]))
        
        f0_valid = np.concatenate((f0_valid, pitch_B_valid[i:i+1].reshape(1,-1)), axis=0)
        f0_input = np.concatenate((f0_input, pitch_A_valid[i:i+1].reshape(1,-1)), axis=0)
    
    del pred_f0, pred_mfc, mfc_source, mfc_target, cyc_pred_f0, cyc_pred_mfc

#    mfc_B_valid = ae_model.get_mfcc(embeddings=mfc_B_valid)
#    mfc_B_valid = np.transpose(mfc_B_valid, (0,2,1))
#    mfc_B_valid = [np.asarray(np.copy(m, order='C'), np.float64) for m in mfc_B_valid]
#    spect_B_valid = [pw.decode_spectral_envelope(m, 16000, 1024) for m in mfc_B_valid]
#    
#    mfc_conv = np.expand_dims(mfc_conv, axis=-1)
#    mfc_conv = np.transpose(mfc_conv, (0,2,1))
#    mfc_conv = ae_model.get_mfcc(embeddings=mfc_conv)
#    mfc_conv = np.transpose(mfc_conv, (0,2,1))
#    mfc_conv = [np.asarray(np.copy(m, order='C'), np.float64) for m in mfc_conv]
#    spect_conv = [pw.decode_spectral_envelope(m, 16000, 1024) for m in mfc_conv]
#        
#    for i in range(10):
#        q = np.random.randint(448)
#        pylab.figure(), pylab.subplot(121)
#        pylab.imshow(_power_to_db(spect_B_valid[q].T ** 2)), pylab.title('Target')
#        pylab.subplot(122)    
#        pylab.imshow(_power_to_db(spect_conv[q].T ** 2)), pylab.title('Converted')
#        pylab.suptitle('Slice %d' % q)



