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
from utils.model_utils import delta_matrix
from nn_models.model_separate_discriminate_id import VariationalCycleGAN
from mfcc_spect_analysis_VCGAN import _power_to_db


num_mfcc = 23
num_pitch = 1
sampling_rate = 16000
frame_period = 5.0


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


if __name__ == '__main__':
    data_valid = scio.loadmat('/home/ravi/Desktop/spect-pitch-gan/data/neu-ang/valid_5.mat')
    
#    pitch_A_valid = np.expand_dims(data_valid['src_f0_feat'], axis=-1)
#    pitch_B_valid = np.expand_dims(data_valid['tar_f0_feat'], axis=-1)
    pitch_A_valid = np.transpose(data_valid['src_f0_feat'], (0,1,3,2))
    pitch_B_valid = np.transpose(data_valid['tar_f0_feat'], (0,1,3,2))
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
    
    model = VariationalCycleGAN(dim_mfc=num_mfcc, dim_pitch=num_pitch, mode='test')
    model.load(filepath='/home/ravi/Desktop/spect-pitch-gan/model/neu-ang/lp_1e-05_lm_1.0_lmo_1e-06_li_0.5_pre_trained_id/neu-ang_1000.ckpt')
    
    f0_conv = np.empty((0,128))
    f0_valid = np.empty((0,128))
    f0_input = np.empty((0,128))
    mfc_conv = np.empty((0,23,128))
    spect_conv = np.empty((0,513,128))
    spect_valid = np.empty((0,513,128))
    spect_input = np.empty((0, 513, 128))
    
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
        
        mfc_source = np.transpose(np.squeeze(mfc_A_valid[i]))
        mfc_source = np.asarray(np.copy(mfc_source, order='C'), np.float64)
        
        f0_valid = np.concatenate((f0_valid, pitch_B_valid[i:i+1].reshape(1,-1)), axis=0)
        f0_input = np.concatenate((f0_input, pitch_A_valid[i:i+1].reshape(1,-1)), axis=0)
        
        spect_target = preproc.world_decode_spectral_envelope(coded_sp=mfc_target, 
                                                       fs=sampling_rate)
        spect_source = preproc.world_decode_spectral_envelope(coded_sp=mfc_source, 
                                                       fs=sampling_rate)
        
        spect_valid = np.concatenate((spect_valid, 
                                      np.expand_dims(spect_target.T, axis=0)), axis=0)
        spect_input = np.concatenate((spect_input, 
                                      np.expand_dims(spect_source.T, axis=0)), axis=0)
        
        q = np.random.uniform(0,1)
        
#        if q < 0.04:
#            pylab.figure()
#            pylab.subplot(131)
#            pylab.imshow(_power_to_db(spect_source.T ** 2)), pylab.title('Source Spect')
#            pylab.subplot(132)
#            pylab.imshow(_power_to_db(spect_target.T ** 2)), pylab.title('Target Spect')
#            pylab.subplot(133)
#            pylab.imshow(_power_to_db(pred_spect.T ** 2)), pylab.title('Predicted Spect')
#            pylab.suptitle('Example %d' % i)
#            pylab.savefig('/home/ravi/Desktop/spect_'+str(i)+'.png')
#            pylab.close()
    
    del pred_f0, pred_mfc, mfc_target, pred_spect, spect_target
    
    
        
    z = delta_matrix()
    mfc_B_valid[np.where(mfc_B_valid==0)] = 1e-10
    mfc_conv[np.where(mfc_conv==0)] = 1e-10
    
    mfc_B_valid_delta = np.dot(mfc_B_valid, z)
    mfc_conv_delta = np.dot(mfc_conv, z)
        
    spect_valid_delta = np.dot(spect_valid, z)
    spect_conv_delta = np.dot(spect_conv, z)
        
#    for i in range(10):
#        q = np.random.randint(448)
#        pylab.figure(), pylab.subplot(121), pylab.imshow(_power_to_db(np.squeeze(mfc_B_valid_delta[q,:,:] ** 2)))
#        pylab.subplot(122), pylab.imshow(_power_to_db(np.squeeze(mfc_conv_delta[q,:,:] ** 2)))
#        pylab.suptitle('slice %d' % q), pylab.savefig('/home/ravi/Desktop/mfcc_grad_'+str(i)+'.png'), pylab.close()
        
#    for i in range(10):
#        q = np.random.randint(448)
#        pylab.figure(), pylab.subplot(121), pylab.imshow(_power_to_db(np.squeeze(spect_valid_delta[q,:,:] ** 2))), pylab.title('Spect Valid')
#        pylab.subplot(122), pylab.imshow(_power_to_db(np.squeeze(spect_conv_delta[q,:,:] ** 2))), pylab.title('Spect Conv')
#        pylab.suptitle('slice %d' % q), pylab.savefig('/home/ravi/Desktop/spect_grad_'+str(i)+'.png'), pylab.close()

##########################################################################################################################
    """
    PCA analysis
    """
    
    import sklearn
    from sklearn.preprocessing import StandardScaler
    
    data_train = scio.loadmat('/home/ravi/Desktop/spect-pitch-gan/data/neu-ang/train_5.mat')
    pitch_A_train = np.transpose(data_train['src_f0_feat'], (0,1,3,2))
    pitch_B_train = np.transpose(data_train['tar_f0_feat'], (0,1,3,2))
    f0_source = np.squeeze(np.vstack(pitch_A_train))
    f0_target = np.squeeze(np.vstack(pitch_B_train))
    
    pca_source = sklearn.decomposition.PCA(n_components=64)
    pca_target = sklearn.decomposition.PCA(n_components=64)
    pca_source.fit(f0_source)
    pca_target.fit(f0_target)
    
    scaler = StandardScaler()
    f0_conv = scaler.fit_transform(f0_conv)
    dist_source = [[np.linalg.norm(x.reshape(-1,) - y.reshape(-1,)) for x in pca_source.components_] for y in f0_conv]
    dist_source = [np.mean(d) for d in dist_source]
    dist_target = [[np.linalg.norm(x.reshape(-1,) - y.reshape(-1,)) for x in pca_target.components_] for y in f0_conv]
    dist_target = [np.mean(d) for d in dist_target]
    pylab.boxplot([dist_source, dist_target], labels=['source dist', 'target dist'])
    pylab.grid()
    
    f0_conv = scaler.inverse_transform(f0_conv)
    
    for i in range(10):
        q = np.random.randint(64)
        pylab.figure(), pylab.plot(pca_source.components_[q,:].reshape(-1,), label='source')
        pylab.plot(pca_target.components_[q,:].reshape(-1,), label='target')
        pylab.legend(), pylab.suptitle('Component %d' % q)





















