from glob import glob
import os
import scipy.io.wavfile as scwav
import numpy as np
import librosa
import scipy.io as scio
import scipy.signal as scisig
import pyworld as pw
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from functools import partial

import warnings
warnings.filterwarnings('ignore')

from feat_utils import smooth, smooth_contour, \
    normalize_wav, encode_raw_spectrum, preprocess_contour


def process_wavs(wav_src, wav_tar, sample_rate=16000, n_feats=128, 
                 n_mfc=23, num_samps=10, window_len=0.005, 
                 window_stride=0.005, encode_raw_spect=False):

    """
    Utterance level features for context expansion
    """
    utt_f0_src          = list()
    utt_f0_tar          = list()
    utt_ec_src          = list()
    utt_ec_tar          = list()
    utt_mfc_src         = list()
    utt_mfc_tar         = list()
    utt_spect_src       = list()
    utt_spect_tar       = list()
    
    file_id = int(wav_src.split('/')[-1][:-4])
    try:
        src_wav = scwav.read(wav_src)
        src = np.asarray(src_wav[1], np.float64)

        tar_wav = scwav.read(wav_tar)
        tar = np.asarray(tar_wav[1], np.float64)
        
        src = normalize_wav(src, floor=-1, ceil=1)
        tar = normalize_wav(tar, floor=-1, ceil=1)

        f0_src, t_src   = pw.harvest(src, sample_rate, frame_period=int(1000*window_len))
        src_straight    = pw.cheaptrick(src, f0_src, t_src, sample_rate)
        src_stft        = np.transpose(librosa.core.stft(src, n_fft=512, 
                                            hop_length=int(window_len*sample_rate), 
                                            win_length=int(0.025*sample_rate)))

        f0_tar, t_tar   = pw.harvest(tar, sample_rate,frame_period=int(1000*window_len))
        tar_straight    = pw.cheaptrick(tar, f0_tar, t_tar, sample_rate)
        tar_stft        = np.transpose(librosa.core.stft(tar, n_fft=512, 
                                            hop_length=int(window_len*sample_rate), 
                                            win_length=int(0.025*sample_rate)))
        
        if encode_raw_spect:
            src_mfc = encode_raw_spectrum(src_straight, axis=1, dim_mfc=n_mfc)
            tar_mfc = encode_raw_spectrum(tar_straight, axis=1, dim_mfc=n_mfc)
        else:
            src_mfc = pw.code_spectral_envelope(src_straight, sample_rate, n_mfc)
            tar_mfc = pw.code_spectral_envelope(tar_straight, sample_rate, n_mfc)
        
        ec_src = np.sqrt(np.sum(np.square(src_straight), axis=1))
        ec_tar = np.sqrt(np.sum(np.square(tar_straight), axis=1))
        
        f0_src = preprocess_contour(f0_src)
        f0_tar = preprocess_contour(f0_tar)
#        ec_src = preprocess_contour(ec_src)
#        ec_tar = preprocess_contour(ec_tar)
        
        f0_src = f0_src.reshape(-1,1)
        f0_tar = f0_tar.reshape(-1,1)

        ec_src = ec_src.reshape(-1,1)
        ec_tar = ec_tar.reshape(-1,1)

        min_length = min([len(f0_src), len(f0_tar)])
        if min_length<n_feats:
            return None
        else:
            for sample in range(num_samps):
                start = np.random.randint(0, min_length-n_feats+1)
                end = start + n_feats
                
                utt_f0_src.append(f0_src[start:end,:])
                utt_f0_tar.append(f0_tar[start:end,:])
                
                utt_ec_src.append(ec_src[start:end,:])
                utt_ec_tar.append(ec_tar[start:end,:])
                
                utt_mfc_src.append(src_mfc[start:end,:])
                utt_mfc_tar.append(tar_mfc[start:end,:])
                
                utt_spect_src.append(src_stft[start:end,:])
                utt_spect_tar.append(tar_stft[start:end,:])
    
        return utt_mfc_src, utt_mfc_tar, utt_f0_src, utt_f0_tar, \
                utt_ec_src, utt_ec_tar, file_id, utt_spect_src, \
                utt_spect_tar

    except Exception as ex:
        template = "An exception of type {0} occurred. Arguments:\n{1!r}"
        message = template.format(type(ex).__name__, ex.args)
        print(message)
        return None
    
    


def get_feats(FILE_LIST, sample_rate, window_len, 
              window_stride, n_feats=128, n_mfc=23, num_samps=10):
    """ 
    FILE_LIST: A list containing the source (first) and target (second) utterances location
    sample_rate: Sampling frequency of the speech
    window_len: Length of the analysis window for getting features (in ms)
    """
    FILE_LIST_src = FILE_LIST[0]
    FILE_LIST_tar = FILE_LIST[1]

    f0_feat_src = list()
    f0_feat_tar = list()

    ec_feat_src = list()
    ec_feat_tar = list()
    
    mfc_feat_src = list()
    mfc_feat_tar = list()
    
    spect_feat_src = list()
    spect_feat_tar = list()

    file_list   = list()
    
    executor = ProcessPoolExecutor(max_workers=6)
    futures = []

    for s,t in zip(FILE_LIST_src, FILE_LIST_tar):
        print(t)
        futures.append(executor.submit(partial(process_wavs, s, t, 
                                               num_samps=num_samps, 
                                               encode_raw_spect=False)))
    
    results = [future.result() for future in tqdm(futures)]
    
    for i in range(len(results)):
        result = results[i]
        try:
            mfc_feat_src.append(result[0])
            mfc_feat_tar.append(result[1])
            
            f0_feat_src.append(result[2])
            f0_feat_tar.append(result[3])
            
            ec_feat_src.append(result[4])
            ec_feat_tar.append(result[5])
            
            file_list.append(result[6])
            
            spect_feat_src.append(result[7])
            spect_feat_tar.append(result[8])
            
        except TypeError:
            print(FILE_LIST_src[i] + " has less than 128 frames.")

    file_list = np.asarray(file_list).reshape(-1,1)
    return file_list, (f0_feat_src, ec_feat_src, mfc_feat_src, \
                       f0_feat_tar, ec_feat_tar, mfc_feat_tar, \
                       spect_feat_src, spect_feat_tar)


##----------------------------------generate CMU-ARCTIC features---------------------------------
#if __name__=='__main__':
#   
#   FILE_LIST_src = sorted(glob(os.path.join('/home/ravi/Desktop/CMU-ARCTIC-US/train/source/', '*.wav')))
#   FILE_LIST_tar = sorted(glob(os.path.join('/home/ravi/Desktop/CMU-ARCTIC-US/train/target/', '*.wav')))
#   
#   sample_rate = 16000.0
#   window_len = 0.005
#   window_stride = 0.005
#   
#   FILE_LIST = [FILE_LIST_src, FILE_LIST_tar]
#   
#   file_names, (src_f0_feat, src_log_f0_feat, src_ec_feat, src_mfc_feat, \
#             tar_f0_feat, tar_log_f0_feat, tar_ec_feat, tar_mfc_feat) \
#             = get_feats(FILE_LIST, sample_rate, window_len, 
#                         window_stride, n_feats=128, n_mfc=23, num_samps=10)
#
#   scio.savemat('./data/cmu_arctic.mat', \
#                { \
#                     'src_mfc_feat':           src_mfc_feat, \
#                     'tar_mfc_feat':           tar_mfc_feat, \
#                     'src_f0_feat':            src_f0_feat, \
#                     'tar_f0_feat':            tar_f0_feat, \
#                     'file_names':             file_names
#                 })
#
#   del file_names, src_mfc_feat, src_f0_feat, src_log_f0_feat, src_ec_feat, \
#       tar_mfc_feat, tar_f0_feat, tar_log_f0_feat, tar_ec_feat


##---------------------------generate VESUS features-------------------------------------------
if __name__=='__main__':
    file_name_dict = {}
    target_emo = 'angry'
    emo_dict = {'neutral-angry':'neu-ang', 'neutral-happy':'neu-hap', \
                'neutral-sad':'neu-sad'}
   
    for i in ['test', 'valid', 'train']:
   
        FILE_LIST_src = sorted(glob(os.path.join('/home/ravi/Downloads/Emo-Conv/', \
                                                 'neutral-'+target_emo+'/'+i+'/neutral/', '*.wav')))
        FILE_LIST_tar = sorted(glob(os.path.join('/home/ravi/Downloads/Emo-Conv/', \
                                                 'neutral-'+target_emo+'/'+i+'/'+target_emo+'/', '*.wav')))
        weights = scio.loadmat('/home/ravi/Downloads/Emo-Conv/neutral-' \
                               +target_emo+'/emo_weight.mat')
       
        sample_rate = 16000.0
        window_len = 0.005
        window_stride = 0.005
       
        FILE_LIST = [FILE_LIST_src, FILE_LIST_tar]
       
        file_names, (src_f0_feat, src_ec_feat, src_mfc_feat, \
                     tar_f0_feat, tar_ec_feat, tar_mfc_feat, \
                     src_spect_feat, tar_spect_feat) \
                     = get_feats(FILE_LIST, sample_rate, window_len, 
                        window_stride, n_feats=128, n_mfc=23, num_samps=30)

        scio.savemat('/home/ravi/Desktop/'+emo_dict['neutral-'+target_emo]+'_unaligned_'+i+'.mat', \
                    { \
                         'src_mfc_feat':   np.asarray(src_mfc_feat, np.float32), \
                         'tar_mfc_feat':   np.asarray(tar_mfc_feat, np.float32), \
                         'src_ec_feat':    np.asarray(src_ec_feat, np.float32), \
                         'tar_ec_feat':    np.asarray(tar_ec_feat, np.float32), \
                         'src_f0_feat':    np.asarray(src_f0_feat, np.float32), \
                         'tar_f0_feat':    np.asarray(tar_f0_feat, np.float32), \
                         'file_names':     file_names
                     })
        
#        scio.savemat('/home/ravi/Desktop/'+emo_dict['neutral-'+target_emo]+'_'+i+'_spect.mat', \
#                    { \
#                         'src_spect_feat':   np.asarray(src_spect_feat, np.float32), \
#                         'tar_spect_feat':   np.asarray(tar_spect_feat, np.float32), \
#                         'file_names':     file_names
#                     })

        file_name_dict[i] = file_names

        del file_names, src_mfc_feat, src_f0_feat, tar_mfc_feat, tar_f0_feat, \
            src_spect_feat, tar_spect_feat





