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

from utils.feat_utils import smooth, \
    smooth_contour, generate_interpolation, 


def process_wavs(wav_src, wav_tar, n_feats=128, n_mfc=23, num_samps=10):

    """
    Utterance level features for context expansion
    """
    utt_log_f0_src      = list()
    utt_log_f0_tar      = list()
    utt_f0_src          = list()
    utt_f0_tar          = list()
    utt_ec_src          = list()
    utt_ec_tar          = list()
    utt_mfc_src         = list()
    utt_mfc_tar         = list()
    
    file_id = int(wav_src.split('/')[-1][:-4])
    try:
        src_wav = scwav.read(wav_src)
        src = np.asarray(src_wav[1], np.float64)

        tar_wav = scwav.read(wav_tar)
        tar = np.asarray(tar_wav[1], np.float64)

        f0_src, t_src   = pw.harvest(src, sample_rate, frame_period=int(1000*window_len))
        straight_src    = pw.cheaptrick(src, f0_src, t_src, sample_rate)

        f0_tar, t_tar   = pw.harvest(tar, sample_rate,frame_period=int(1000*window_len))
        straight_tar    = pw.cheaptrick(tar, f0_tar, t_tar, sample_rate)

        f0_src = scisig.medfilt(f0_src, kernel_size=3)
        f0_tar = scisig.medfilt(f0_tar, kernel_size=3)
        f0_src = np.asarray(f0_src, np.float32)
        f0_tar = np.asarray(f0_tar, np.float32)

        ec_src = np.sqrt(np.sum(np.square(straight_src), axis=1))
        ec_tar = np.sqrt(np.sum(np.square(straight_tar), axis=1))
        ec_src = scisig.medfilt(ec_src, kernel_size=3)
        ec_tar = scisig.medfilt(ec_tar, kernel_size=3)
        ec_src = np.asarray(ec_src, np.float32)
        ec_tar = np.asarray(ec_tar, np.float32)

        f0_src = np.asarray(generate_interpolation(f0_src), np.float32)
        f0_tar = np.asarray(generate_interpolation(f0_tar), np.float32)
        ec_src = np.asarray(generate_interpolation(ec_src), np.float32)
        ec_tar = np.asarray(generate_interpolation(ec_tar), np.float32)
        
        f0_src = smooth(f0_src, window_len=13)
        f0_tar = smooth(f0_tar, window_len=13)
        ec_src = smooth(ec_src, window_len=13)
        ec_tar = smooth(ec_tar, window_len=13)

        src_mfc = pw.code_spectral_envelope(straight_src, sample_rate, n_mfc)
        tar_mfc = pw.code_spectral_envelope(straight_tar, sample_rate, n_mfc)

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
        
        f0_src = f0_src.reshape(-1,1)
        f0_tar = f0_tar.reshape(-1,1)
        
        ec_src = ec_src.reshape(-1,1)
        ec_tar = ec_tar.reshape(-1,1)
        
        ext_src_f0 = []
        ext_tar_f0 = []
        ext_src_ec = []
        ext_tar_ec = []
        ext_src_mfc = []
        ext_tar_mfc = []
        
        for i in range(len(cords)-1, -1, -1):
            ext_src_f0.append(f0_src[cords[i,0],0])
            ext_tar_f0.append(f0_tar[cords[i,1],0])
            ext_src_ec.append(ec_src[cords[i,0],0])
            ext_tar_ec.append(ec_tar[cords[i,1],0])
            ext_src_mfc.append(src_mfc[cords[i,0],:])
            ext_tar_mfc.append(tar_mfc[cords[i,1],:])
        
        ext_src_f0 = np.reshape(np.asarray(ext_src_f0), (-1,1))
        ext_tar_f0 = np.reshape(np.asarray(ext_tar_f0), (-1,1))
        ext_src_ec = np.reshape(np.asarray(ext_src_ec), (-1,1))
        ext_tar_ec = np.reshape(np.asarray(ext_tar_ec), (-1,1))
        ext_log_src_f0 = np.reshape(np.log(np.asarray(ext_src_f0)), (-1,1))
        ext_log_tar_f0 = np.reshape(np.log(np.asarray(ext_tar_f0)), (-1,1))
        ext_src_mfc = np.asarray(ext_src_mfc)
        ext_tar_mfc = np.asarray(ext_tar_mfc)

        if cords.shape[0]<n_feats:
            return None
        else:
            for sample in range(num_samps):
                start = np.random.randint(0, cords.shape[0]-n_feats+1)
                end = start + n_feats
                
                utt_f0_src.append(ext_src_f0[start:end,:])
                utt_f0_tar.append(ext_tar_f0[start:end,:])
                
                utt_log_f0_src.append(ext_log_src_f0[start:end,:])
                utt_log_f0_tar.append(ext_log_tar_f0[start:end,:])
                
                utt_ec_src.append(ext_src_ec[start:end,:])
                utt_ec_tar.append(ext_tar_ec[start:end,:])
                
                utt_mfc_src.append(ext_src_mfc[start:end,:])
                utt_mfc_tar.append(ext_tar_mfc[start:end,:])
        
        return utt_mfc_src, utt_mfc_tar, utt_f0_src, utt_f0_tar, \
                utt_log_f0_src, utt_log_f0_tar, utt_ec_src, utt_ec_tar, file_id

    except Exception as ex:
        template = "An exception of type {0} occurred. Arguments:\n{1!r}"
        message = template.format(type(ex).__name__, ex.args)
        print(message)
        return None
    
    


def get_feats(FILE_LIST, sample_rate, window_len, 
              window_stride, n_feats=128, n_mfc=23):
    """ 
    FILE_LIST: A list containing the source (first) and target (second) utterances location
    sample_rate: Sampling frequency of the speech
    window_len: Length of the analysis window for getting features (in ms)
    """
    FILE_LIST_src = FILE_LIST[0]
    FILE_LIST_tar = FILE_LIST[1]

    f0_feat_src = []
    f0_feat_tar = []
    
    log_f0_feat_src = []
    log_f0_feat_tar = []
    
    ec_feat_src = []
    ec_feat_tar = []
    
    mfc_feat_src = []
    mfc_feat_tar = []

    file_list   = []
    
    executor = ProcessPoolExecutor(max_workers=6)
    futures = []

    for s,t in zip(FILE_LIST_src, FILE_LIST_tar):
        print(t)
        futures.append(executor.submit(partial(process_wavs, s, t, num_samps=15)))
    
    results = [future.result() for future in tqdm(futures)]
    
    for result in results:
        
        mfc_feat_src.append(result[0])
        mfc_feat_tar.append(result[1])
        
        f0_feat_src.append(result[2])
        f0_feat_tar.append(result[3])
        
        log_f0_feat_src.append(result[4])
        log_f0_feat_tar.append(result[5])
        
        ec_feat_src.append(result[6])
        ec_feat_tar.append(result[7])
        
        file_list.append(result[8])

    file_list = np.asarray(file_list).reshape(-1,1)
    return file_list, (f0_feat_src, log_f0_feat_src, ec_feat_src, \
                     mfc_feat_src, f0_feat_tar, log_f0_feat_tar, \
                     ec_feat_tar, mfc_feat_tar)


##----------------------------------generate all features---------------------------------
if __name__=='__main__':
   
   FILE_LIST_src = sorted(glob(os.path.join('/home/ravi/Desktop/CMU-ARCTIC-US/train/source/', '*.wav')))
   FILE_LIST_tar = sorted(glob(os.path.join('/home/ravi/Desktop/CMU-ARCTIC-US/train/target/', '*.wav')))
   
   sample_rate = 16000.0
   window_len = 0.005
   window_stride = 0.005
   
   FILE_LIST = [FILE_LIST_src, FILE_LIST_tar]
   
   file_names, (src_f0_feat, src_log_f0_feat, src_ec_feat, src_mfc_feat, \
             tar_f0_feat, tar_log_f0_feat, tar_ec_feat, tar_mfc_feat) \
             = get_feats(FILE_LIST, sample_rate, window_len, 
                         window_stride, n_feats=128, n_mfc=23)

   scio.savemat('./data/cmu_arctic.mat', \
                { \
                     'src_mfc_feat':           src_mfc_feat, \
                     'tar_mfc_feat':           tar_mfc_feat, \
                     'src_f0_feat':            src_f0_feat, \
                     'tar_f0_feat':            tar_f0_feat, \
                     'file_names':             file_names
                 })

   del file_names, src_mfc_feat, src_f0_feat, src_log_f0_feat, src_ec_feat, \
       tar_mfc_feat, tar_f0_feat, tar_log_f0_feat, tar_ec_feat