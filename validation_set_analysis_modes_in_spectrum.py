import argparse
import os
import numpy as np
import librosa
import scipy.io.wavfile as scwav
import scipy.signal as scisig
import scipy.io as scio
import pylab

import utils.preprocess as preproc
from utils.helper import smooth, generate_interpolation
from nn_models.model_separate_discriminate_id import VariationalCycleGAN


num_mfcc = 23
num_pitch = 1
sampling_rate = 16000
frame_period = 5.0


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


if __name__ == '__main__':
    data_valid = scio.loadmat('/home/ravi/Desktop/spect-pitch-gan/data/neu-ang/valid_5.mat')
    
    pitch_A_valid = np.expand_dims(data_valid['src_f0_feat'], axis=-1)
    pitch_B_valid = np.expand_dims(data_valid['tar_f0_feat'], axis=-1)
    mfc_A_valid = data_valid['src_mfc_feat']
    mfc_B_valid = data_valid['tar_mfc_feat']
    
    mfc_A_valid, pitch_A_valid, \
        mfc_B_valid, pitch_B_valid = preproc.sample_data(mfc_A=mfc_A_valid, \
                                    mfc_B=mfc_B_valid, pitch_A=pitch_A_valid, \
                                    pitch_B=pitch_B_valid)
    
    model = VariationalCycleGAN(dim_mfc=num_mfcc, dim_pitch=num_pitch, mode='test')
    model.load(filepath='/home/ravi/Desktop/spect-pitch-gan/model/neu-ang/lp_1e-05_lm_1.0_lmo_1e-06_li_0.5_pre_trained_id/neu-ang_1000.ckpt')
    
    f0_conv = np.empty((0,128))
    mfc_conv = np.empty((0,23,128))
    spect_conv = np.empty((0,513,128))
    spect_valid = np.empty((0,513,128))
    
    for i in range(mfc_A_valid.shape[0]):
                
        pred_f0, pred_mfc = model.test(input_pitch=pitch_A_valid[i:i+1], 
                                           input_mfc=mfc_A_valid[i:i+1], 
                                           direction='A2B')
        f0_conv = np.concatenate((f0_conv, pred_f0.reshape(1,-1)), axis=0)
        mfc_conv = np.concatenate((mfc_conv, pred_mfc), axis=0)
        pred_mfc = np.asarray(np.squeeze(pred_mfc), np.float64)
        pred_mfc = np.copy(pred_mfc.T, order='C')
        pred_spect = preproc.world_decode_spectral_envelope(coded_sp=pred_mfc, 
                                                            fs=sampling_rate)
        spect_conv = np.concatenate((spect_conv, 
                                     np.expand_dims(pred_spect.T, axis=0)), axis=0)
        
        mfc_target = np.transpose(np.squeeze(mfc_B_valid[i]))
        mfc_target = np.asarray(np.copy(mfc_target, order='C'), np.float64)
        spect_target = preproc.world_decode_spectral_envelope(coded_sp=mfc_target, 
                                                       fs=sampling_rate)
        spect_valid = np.concatenate((spect_valid, 
                                      np.expand_dims(spect_target.T, axis=0)), axis=0)
        
        q = np.random.randint(0, 128)
        
        pylab.figure()
        pylab.plot(spect_target[q,:].reshape(-1,), 'g', label='Target Spect')
        pylab.plot(pred_spect[q,:].reshape(-1,), 'r', label='Generated Spect')
        pylab.legend(loc=1), pylab.title('Slice %d' % q)
        pylab.savefig('/home/ravi/Desktop/'+str(i)+'_'+str(q)+'.png')
        pylab.close()
    
    del pred_f0, pred_mfc, mfc_target, pred_spect, spect_target



