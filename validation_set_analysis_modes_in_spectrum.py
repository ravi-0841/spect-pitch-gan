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
#from nn_models.model_separate_discriminate_id import VariationalCycleGAN
from nn_models.model_wasserstein import VariationalCycleGAN
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
#    model.load(filepath='./model/neu-ang/lp_1e-05_lm_1.0_lmo_1e-06_li_0.5_pre_trained_id_3500/neu-ang_3500.ckpt')
#    model.load(filepath='./model/neu-ang/lp_1e-05_lm_1.0_lmo_1e-06_li_0.5_pre_trained_id_1000/neu-ang_1000.ckpt')
#    model.load(filepath='./model/neu-ang/lp_1e-05_lm_0.1_lmo_1e-06_li_0.05_glr1e-07_dlr_1e-07_pre_trained_spect_loss_inv_norm/neu-ang_1200.ckpt')
#    model.load(filepath='./model/neu-ang/lp_1e-05_lm_0.1_lmo_1e-06_li_0.05_glr1e-07_dlr_1e-07_pre_trained_spect_loss/neu-ang_700.ckpt')
    model.load(filepath='./model/neu-ang/lp_1e-05_lm_1.0_lmo_1e-06_li_0.5_wasserstein/neu-ang_2100.ckpt')
    
    f0_conv = np.empty((0,128))
    f0_valid = np.empty((0,128))
    f0_input = np.empty((0,128))
    cyc_f0 = np.empty((0,128))
    mfc_conv = np.empty((0,23,128))
    cyc_mfc = np.empty((0,23,128))
    spect_conv = np.empty((0,513,128))
    spect_output = np.empty((0,513,128))
    spect_input = np.empty((0, 513, 128))
    cyc_spect = np.empty((0, 513, 128))

    for i in range(mfc_A_valid.shape[0]):

        pred_f0, pred_mfc = model.test(input_pitch=pitch_A_valid[i:i+1], 
                                           input_mfc=mfc_A_valid[i:i+1], 
                                           direction='A2B')
        cyc_pred_f0, cyc_pred_mfc = model.test(input_pitch=pred_f0, 
                                               input_mfc=pred_mfc, 
                                               direction='B2A')
        
        f0_conv = np.concatenate((f0_conv, pred_f0.reshape(1,-1)), axis=0)
        cyc_f0 = np.concatenate((cyc_f0, cyc_pred_f0.reshape(1,-1)), axis=0)
        
        mfc_conv = np.concatenate((mfc_conv, pred_mfc), axis=0)
        pred_mfc = np.asarray(np.squeeze(pred_mfc), np.float64)
        pred_mfc = np.copy(pred_mfc.T, order='C')
        pred_spect = preproc.world_decode_spectral_envelope(coded_sp=pred_mfc, 
                                                            fs=sampling_rate)
        
        spect_conv = np.concatenate((spect_conv, 
                                     np.expand_dims(pred_spect.T, axis=0)), axis=0)
        
        cyc_mfc = np.concatenate((cyc_mfc, cyc_pred_mfc), axis=0)
        cyc_pred_mfc = np.asarray(np.squeeze(cyc_pred_mfc), np.float64)
        cyc_pred_mfc = np.copy(cyc_pred_mfc.T, order='C')
        cyc_pred_spect = preproc.world_decode_spectral_envelope(coded_sp=cyc_pred_mfc, 
                                                            fs=sampling_rate)
        
        cyc_spect = np.concatenate((cyc_spect, 
                                     np.expand_dims(cyc_pred_spect.T, axis=0)), axis=0)
        
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
        
        spect_output = np.concatenate((spect_output, 
                                      np.expand_dims(spect_target.T, axis=0)), axis=0)
        spect_input = np.concatenate((spect_input, 
                                      np.expand_dims(spect_source.T, axis=0)), axis=0)
        
        q = np.random.uniform(0,1)
        
#        if q < 0.03:
#            pylab.figure(figsize=(13,13))
#            pylab.subplot(131)
#            pylab.imshow(_power_to_db(spect_source.T ** 2)), pylab.title('Source Spect')
#            pylab.subplot(132)
#            pylab.imshow(_power_to_db(spect_target.T ** 2)), pylab.title('Target Spect')
#            pylab.subplot(133)
#            pylab.imshow(_power_to_db(pred_spect.T ** 2)), pylab.title('Predicted Spect')
#            pylab.suptitle('Example %d' % i)
#            pylab.savefig('/home/ravi/Desktop/spect_'+str(i)+'.png')
#            pylab.close()
    
    del pred_f0, pred_mfc, mfc_target, pred_spect, spect_target, \
        cyc_pred_f0, cyc_pred_mfc, cyc_pred_spect
    
    
        
    z = delta_matrix()
    mfc_B_valid[np.where(mfc_B_valid==0)] = 1e-10
    mfc_conv[np.where(mfc_conv==0)] = 1e-10
    
    mfc_B_valid_delta = np.dot(mfc_B_valid, z)
    mfc_conv_delta = np.dot(mfc_conv, z)
        
    spect_output_delta = np.dot(spect_output, z)
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
    
#    for i in range(10):
#        q = np.random.randint(0,448)
#        pylab.figure(figsize=(15,15))
#        pylab.subplot(141), pylab.imshow(normalize(_power_to_db(np.squeeze(spect_input[q,:,:]) ** 2))), 
#        pylab.title('Input Spect'), pylab.colorbar()
#        pylab.subplot(142), pylab.imshow(normalize(_power_to_db(np.squeeze(cyc_spect[q,:,:]) ** 2))), 
#        pylab.title('Cyclic Spect'), pylab.colorbar()
#        pylab.subplot(143), pylab.imshow(normalize(_power_to_db(np.squeeze(spect_conv[q,:,:]) ** 2))), 
#        pylab.title('Conv Spect'), pylab.colorbar()
#        pylab.subplot(144), pylab.imshow(normalize(_power_to_db(np.squeeze(spect_output[q,:,:]) ** 2))), 
#        pylab.title('Target Spect'), pylab.colorbar()
#        pylab.suptitle('Example %d' % q)
#        pylab.savefig('/home/ravi/Desktop/spect_consistency_'+str(i)+'.png')
#        pylab.close()


##########################################################################################################################
    """
    PCA analysis
    """
    
#    import sklearn
#    from sklearn.preprocessing import StandardScaler
#    
#    data_train = scio.loadmat('/home/ravi/Desktop/spect-pitch-gan/data/neu-ang/train_5.mat')
#    pitch_A_train = np.transpose(data_train['src_f0_feat'], (0,1,3,2))
#    pitch_B_train = np.transpose(data_train['tar_f0_feat'], (0,1,3,2))
#    f0_source = np.squeeze(np.vstack(pitch_A_train))
#    f0_target = np.squeeze(np.vstack(pitch_B_train))
#    
#    pca_source = sklearn.decomposition.PCA(n_components=64)
#    pca_target = sklearn.decomposition.PCA(n_components=64)
#    pca_source.fit(f0_source)
#    pca_target.fit(f0_target)
#    
#    scaler = StandardScaler()
#    f0_conv = scaler.fit_transform(f0_conv)
#    dist_source = [[np.linalg.norm(x.reshape(-1,) - y.reshape(-1,)) for x in pca_source.components_] for y in f0_conv]
#    dist_source = [np.mean(d) for d in dist_source]
#    dist_target = [[np.linalg.norm(x.reshape(-1,) - y.reshape(-1,)) for x in pca_target.components_] for y in f0_conv]
#    dist_target = [np.mean(d) for d in dist_target]
#    pylab.boxplot([dist_source, dist_target], labels=['source dist', 'target dist'])
#    pylab.grid()
#    
#    f0_conv = scaler.inverse_transform(f0_conv)
#    
#    for i in range(10):
#        q = np.random.randint(64)
#        pylab.figure(), pylab.plot(pca_source.components_[q,:].reshape(-1,), label='source')
#        pylab.plot(pca_target.components_[q,:].reshape(-1,), label='target')
#        pylab.legend(), pylab.suptitle('Component %d' % q)


###############################################################################################################################
    """
    Sparse-Dense decomposition of Mfcc matrix
    """
#    kernel_np = model.sess.run(model.generator_vars)
#    A2B_h1 = kernel_np[62]
#    for i in range(64):
#        pylab.figure(figsize=(13,13))
#        inv_filt = scipy.fftpack.idct(np.squeeze(A2B_h1[:,:-1,i]), axis=-1, n=65)
#        pylab.subplot(121)
#        pylab.imshow(np.squeeze(A2B_h1[:,:-1,i]))
#        pylab.title('MFCC Kernel %d' % i)
#        pylab.subplot(122)
#        pylab.imshow(inv_filt.T)
#        pylab.title('IDCT Kernel %d' % i)
#        pylab.savefig('/home/ravi/Desktop/mfcc_generator_kernel_1/kernel_'+str(i)+'.png')
#        pylab.close()
    
#    projection_mat = np.random.randn(23, 23)
#    projection_mat = sym(projection_mat)
#    projection_mat_inv = np.linalg.inv(projection_mat)
#    mfc_proj_A = [np.dot(np.transpose(np.squeeze(x)), projection_mat) for x in mfc_A_valid]
#    mfc_inv_proj_A = [np.dot(x, projection_mat_inv) for x in mfc_proj_A]
#    
#    mfc_proj_B = [np.dot(np.transpose(np.squeeze(x)), projection_mat) for x in mfc_B_valid]
#    mfc_inv_proj_B = [np.dot(x, projection_mat_inv) for x in mfc_proj_B]
#    
#    for i in range(10):
#        q = np.random.randint(448)
#        pylab.figure()
#        pylab.subplot(131), pylab.imshow(_power_to_db(np.transpose(np.squeeze(mfc_A_valid[q])) ** 2)), pylab.title('Original')
#        pylab.subplot(132), pylab.imshow(_power_to_db(mfc_proj_A[q] ** 2)), pylab.title('Projected')
#        pylab.subplot(133), pylab.imshow(_power_to_db(mfc_inv_proj_A[q] ** 2)), pylab.title('Inverted')
#        pylab.suptitle('Slice %d' % q)
        
    
    



















