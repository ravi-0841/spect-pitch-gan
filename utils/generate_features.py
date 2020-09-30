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


def preprocess_pitch(pitch):
    """
    Expects pitch as a numpy array of shape (T,)
    """
    pitch = scisig.medfilt(pitch, kernel_size=3)
    pitch = np.asarray(generate_interpolation(pitch), np.float32)
    pitch = smooth(pitch, window_len=13)
    return pitch


def process_wavs(wav_src, wav_tar, sample_rate=16000, n_feats=128, 
                 n_mfc=23, num_samps=10, window_len=0.005, 
                 window_stride=0.005, encode_raw_spect=False):

    """
    Utterance level features for context expansion
    """
#    utt_log_f0_src      = list()
#    utt_log_f0_tar      = list()
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
        
        src = normalize_wav(src)
        tar = normalize_wav(tar)

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
            
        ec_src = np.sqrt(np.sum(np.square(src_mfc), axis=1))
        ec_tar = np.sqrt(np.sum(np.square(tar_mfc), axis=1))
        
        f0_src = preprocess_contour(f0_src)
        f0_tar = preprocess_contour(f0_tar)
        ec_src = preprocess_contour(ec_src)
        ec_tar = preprocess_contour(ec_tar)
        
        f0_src = f0_src.reshape(-1,1)
        f0_tar = f0_tar.reshape(-1,1)

        ec_src = ec_src.reshape(-1,1)
        ec_tar = ec_tar.reshape(-1,1)

        src_mfcc = librosa.feature.mfcc(y=src, sr=sample_rate, \
                                        hop_length=int(sample_rate*window_len), \
                                        win_length=int(sample_rate*window_len), \
                                        n_fft=1024, n_mels=128)
        
        tar_mfcc = librosa.feature.mfcc(y=tar, sr=sample_rate, \
                                        hop_length=int(sample_rate*window_len), \
                                        win_length=int(sample_rate*window_len), \
                                        n_fft=1024, n_mels=128)

        _, cords = librosa.sequence.dtw(X=src_mfcc, Y=tar_mfcc, metric='cosine')

        del src_mfcc, tar_mfcc
        
        ext_src_f0 = list()
        ext_tar_f0 = list()
        ext_src_ec = list()
        ext_tar_ec = list()
        ext_src_mfc = list()
        ext_tar_mfc = list()
        ext_src_spect = list()
        ext_tar_spect = list()
        
        for i in range(len(cords)-1, -1, -1):
            ext_src_f0.append(f0_src[cords[i,0],0])
            ext_tar_f0.append(f0_tar[cords[i,1],0])
            ext_src_ec.append(ec_src[cords[i,0],0])
            ext_tar_ec.append(ec_tar[cords[i,1],0])
            ext_src_mfc.append(src_mfc[cords[i,0],:])
            ext_tar_mfc.append(tar_mfc[cords[i,1],:])
            ext_src_spect.append(src_stft[cords[i,0],:])
            ext_tar_spect.append(tar_stft[cords[i,1],:])
        
        ext_src_f0 = np.reshape(np.asarray(ext_src_f0), (-1,1))
        ext_tar_f0 = np.reshape(np.asarray(ext_tar_f0), (-1,1))
        ext_src_ec = np.reshape(np.asarray(ext_src_ec), (-1,1))
        ext_tar_ec = np.reshape(np.asarray(ext_tar_ec), (-1,1))
#        ext_log_src_f0 = np.reshape(np.log(np.asarray(ext_src_f0)), (-1,1))
#        ext_log_tar_f0 = np.reshape(np.log(np.asarray(ext_tar_f0)), (-1,1))
        ext_src_mfc = np.asarray(ext_src_mfc)
        ext_tar_mfc = np.asarray(ext_tar_mfc)
        ext_src_spect = np.asarray(ext_src_spect)
        ext_tar_spect = np.asarray(ext_tar_spect)
        
        src_mfc = np.asarray(src_mfc, np.float32)
        tar_mfc = np.asarray(tar_mfc, np.float32)
        src_stft = np.asarray(src_stft, np.float32)
        tar_stft = np.asarray(tar_stft, np.float32)

        if cords.shape[0]<n_feats:
            return None
        else:
            for sample in range(num_samps):
                start = np.random.randint(0, cords.shape[0]-n_feats+1)
                end = start + n_feats
                
                utt_f0_src.append(ext_src_f0[start:end,:])
                utt_f0_tar.append(ext_tar_f0[start:end,:])
                
#                utt_log_f0_src.append(ext_log_src_f0[start:end,:])
#                utt_log_f0_tar.append(ext_log_tar_f0[start:end,:])
                
                utt_ec_src.append(ext_src_ec[start:end,:])
                utt_ec_tar.append(ext_tar_ec[start:end,:])
                
                utt_mfc_src.append(ext_src_mfc[start:end,:])
                utt_mfc_tar.append(ext_tar_mfc[start:end,:])
                
                utt_spect_src.append(ext_src_spect[start:end,:])
                utt_spect_tar.append(ext_tar_spect[start:end,:])
        
        return utt_mfc_src, utt_mfc_tar, utt_f0_src, utt_f0_tar, \
                utt_ec_src, utt_ec_tar, file_id

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
    
#    log_f0_feat_src = list()
#    log_f0_feat_tar = list()
    
    ec_feat_src = list()
    ec_feat_tar = list()
    
    mfc_feat_src = list()
    mfc_feat_tar = list()
    
#    spect_feat_src = list()
#    spect_feat_tar = list()

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
            
#            log_f0_feat_src.append(result[4])
#            log_f0_feat_tar.append(result[5])
            
            ec_feat_src.append(result[4])
            ec_feat_tar.append(result[5])

#            spect_feat_src.append(result[6])
#            spect_feat_tar.append(result[7])

            file_list.append(result[6])
            
        except TypeError:
            print(FILE_LIST_src[i] + " has less than 128 frames.")

    file_list = np.asarray(file_list).reshape(-1,1)
    return file_list, (f0_feat_src, ec_feat_src, mfc_feat_src, \
                       f0_feat_tar, ec_feat_tar, mfc_feat_tar)


##----------------------------------generate CMU-ARCTIC features---------------------------------
if __name__=='__main__':
   
   FILE_LIST_src = sorted(glob(os.path.join('/home/ravi/Desktop/spect-pitch-gan/data/CMU-ARCTIC-US/train/source/', '*.wav')))
   FILE_LIST_tar = sorted(glob(os.path.join('/home/ravi/Desktop/spect-pitch-gan/data/CMU-ARCTIC-US/train/target/', '*.wav')))
   
   sample_rate = 16000.0
   window_len = 0.005
   window_stride = 0.005
   
   FILE_LIST = [FILE_LIST_src, FILE_LIST_tar]
   
   file_names, (src_f0_feat, src_ec_feat, src_mfc_feat, \
             tar_f0_feat, tar_ec_feat, tar_mfc_feat) \
             = get_feats(FILE_LIST, sample_rate, window_len, 
                         window_stride, n_feats=128, n_mfc=23, num_samps=10)

   scio.savemat('/home/ravi/Desktop/mfc_energy_cmu_arctic.mat', \
                { \
                     'src_mfc_feat':           src_mfc_feat, \
                     'tar_mfc_feat':           tar_mfc_feat, \
                     'src_f0_feat':            src_f0_feat, \
                     'tar_f0_feat':            tar_f0_feat, \
                     'src_ec_feat':            src_ec_feat, \
                     'tar_ec_feat':            tar_ec_feat, \
                     'file_names':             file_names
                 })

   del file_names, src_mfc_feat, src_f0_feat, src_ec_feat, \
       tar_mfc_feat, tar_f0_feat, tar_ec_feat


##---------------------------generate VESUS features-------------------------------------------
#if __name__=='__main__':
#    file_name_dict = {}
#    target_emo = 'angry'
#    emo_dict = {'neutral-angry':'neu-ang', 'neutral-happy':'neu-hap', \
#                'neutral-sad':'neu-sad'}
#   
#    for i in ['test_reshuff', 'valid_reshuff', 'train_reshuff']:
#   
#        FILE_LIST_src = sorted(glob(os.path.join('/home/ravi/Downloads/Emo-Conv/', \
#                                                 'neutral-'+target_emo+'/'+i+'/neutral/', '*.wav')))
#        FILE_LIST_tar = sorted(glob(os.path.join('/home/ravi/Downloads/Emo-Conv/', \
#                                                 'neutral-'+target_emo+'/'+i+'/'+target_emo+'/', '*.wav')))
#        weights = scio.loadmat('/home/ravi/Downloads/Emo-Conv/neutral-' \
#                               +target_emo+'/emo_weight.mat')
#       
#        sample_rate = 16000.0
#        window_len = 0.005
#        window_stride = 0.005
#       
#        FILE_LIST = [FILE_LIST_src, FILE_LIST_tar]
#       
#        file_names, (src_f0_feat, src_mfc_feat, tar_f0_feat, tar_mfc_feat, \
#                     src_spect_feat, tar_spect_feat) \
#                     = get_feats(FILE_LIST, sample_rate, window_len, 
#                             window_stride, n_feats=128, n_mfc=23, num_samps=40)
#
#        scio.savemat('/home/ravi/Desktop/'+emo_dict['neutral-'+target_emo]+'_'+i+'.mat', \
#                    { \
#                         'src_mfc_feat':   np.asarray(src_mfc_feat, np.float32), \
#                         'tar_mfc_feat':   np.asarray(tar_mfc_feat, np.float32), \
#                         'src_f0_feat':    np.asarray(src_f0_feat, np.float32), \
#                         'tar_f0_feat':    np.asarray(tar_f0_feat, np.float32), \
#                         'file_names':     file_names
#                     })
        
#        scio.savemat('/home/ravi/Desktop/'+emo_dict['neutral-'+target_emo]+'_'+i+'_spect.mat', \
#                    { \
#                         'src_spect_feat':   np.asarray(src_spect_feat, np.float32), \
#                         'tar_spect_feat':   np.asarray(tar_spect_feat, np.float32), \
#                         'file_names':     file_names
#                     })

#        file_name_dict[i] = file_names
#
#        del file_names, src_mfc_feat, src_f0_feat, tar_mfc_feat, tar_f0_feat, \
#            src_spect_feat, tar_spect_feat





