#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 27 14:39:08 2020

@author: ravi
"""

import scipy.io as scio
import scipy.io.wavfile as scwav
import numpy as np
import joblib
import pyworld as pw
import os
import warnings
warnings.filterwarnings('ignore')

from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from glob import glob
from extract_fold_data_hparams import Hparams
from feat_utils import normalize_wav, preprocess_contour

dict_target_emo = {'neutral_angry':['angry', 'neu-ang'], \
                   'neutral_happy':['happy', 'neu-hap'], \
                   'neutral_sad':['sad', 'neu-sad']}


def _process_wavs(wav_src, wav_tar, args):

    """
    Utterance level features for context expansion
    """
    utt_f0_src          = list()
    utt_f0_tar          = list()
    utt_ec_src          = list()
    utt_ec_tar          = list()
    utt_mfc_src         = list()
    utt_mfc_tar         = list()

    try:
        src_wav = scwav.read(wav_src)
        src = np.asarray(src_wav[1], np.float64)

        tar_wav = scwav.read(wav_tar)
        tar = np.asarray(tar_wav[1], np.float64)
        
        src = normalize_wav(src, floor=-1, ceil=1)
        tar = normalize_wav(tar, floor=-1, ceil=1)

        f0_src, t_src   = pw.harvest(src, args.fs, frame_period=int(1000*args.win_len))
        src_straight    = pw.cheaptrick(src, f0_src, t_src, args.fs)

        f0_tar, t_tar   = pw.harvest(tar, args.fs,frame_period=int(1000*args.win_len))
        tar_straight    = pw.cheaptrick(tar, f0_tar, t_tar, args.fs)

        src_mfc = pw.code_spectral_envelope(src_straight, args.fs, args.n_mels)
        tar_mfc = pw.code_spectral_envelope(tar_straight, args.fs, args.n_mels)

        ec_src = np.sum(src_mfc, axis=1)
        ec_tar = np.sum(tar_mfc, axis=1)
        
        f0_src = preprocess_contour(f0_src)
        f0_tar = preprocess_contour(f0_tar)

        ec_src = preprocess_contour(ec_src)
        ec_tar = preprocess_contour(ec_tar)
        
        f0_src = f0_src.reshape(-1,1)
        f0_tar = f0_tar.reshape(-1,1)

        ec_src = ec_src.reshape(-1,1)
        ec_tar = ec_tar.reshape(-1,1)

        min_length = min([len(f0_src), len(f0_tar)])
        if min_length<args.frame_len:
            return None, None, None, None, None, None, None
        else:
            for sample in range(args.n_samplings):
                start = np.random.randint(0, min_length-args.frame_len+1)
                end = start + args.frame_len
                
                utt_f0_src.append(f0_src[start:end,:])
                utt_f0_tar.append(f0_tar[start:end,:])
                
                utt_ec_src.append(ec_src[start:end,:])
                utt_ec_tar.append(ec_tar[start:end,:])
                
                utt_mfc_src.append(src_mfc[start:end,:])
                utt_mfc_tar.append(tar_mfc[start:end,:])
    
        return utt_mfc_src, utt_mfc_tar, utt_f0_src, utt_f0_tar, \
                utt_ec_src, utt_ec_tar, int(os.path.basename(wav_src)[:-4])

    except Exception as ex:
        template = "An exception of type {0} occurred. Arguments:\n{1!r}"
        message = template.format(type(ex).__name__, ex.args)
        print(message)
        return None, None, None, None, None, None, None


def extract_features(args):

    target = dict_target_emo[args.emo_pair][0]
    speaker_file_info = joblib.load('/home/ravi/Downloads/Emo-Conv/speaker_file_info.pkl')
    speaker_info = speaker_file_info[args.emo_pair]
    test_speaker_female = speaker_info[args.fold - 1]
    test_speaker_male = speaker_info[5 + args.fold - 1]
    
    test_file_idx = [i for i in range(test_speaker_female[0], test_speaker_female[1]+1)]
    test_file_idx += [i for i in range(test_speaker_male[0], test_speaker_male[1]+1)]
    
    src_files = sorted(glob(os.path.join(args.data_folder, 'neutral', '*.wav')))
    tar_files = sorted(glob(os.path.join(args.data_folder, target, '*.wav')))
    
    train_f0_src = list()
    train_f0_tar = list()
    train_ec_src = list()
    train_ec_tar = list()
    train_mfc_src = list()
    train_mfc_tar = list()
    
    valid_f0_src = list()
    valid_f0_tar = list()
    valid_ec_src = list()
    valid_ec_tar = list()
    valid_mfc_src = list()
    valid_mfc_tar = list()
    
    test_f0_src = list()
    test_f0_tar = list()
    test_ec_src = list()
    test_ec_tar = list()
    test_mfc_src = list()
    test_mfc_tar = list()
    
    train_files = list()
    valid_files = list()
    test_files = list()
    
    executor = ProcessPoolExecutor(max_workers=6)
    futures = []
    
    for (s,t) in zip(src_files, tar_files):
        print("Processing: {0}".format(s))
        futures.append(executor.submit(partial(_process_wavs, s, t, 
                                               args=args)))
    
    results = [future.result() for future in tqdm(futures)]
    
    for i in range(len(results)):
        result = results[i]
        mfc_src = result[0]
        mfc_tar = result[1]
        f0_src  = result[2]
        f0_tar  = result[3]
        ec_src  = result[4]
        ec_tar  = result[5]
        file_idx= result[6]
    
#        mfc_src, mfc_tar, f0_src, \
#            f0_tar, ec_src, ec_tar, file_idx = _process_wavs(s,t,args)
    
        if mfc_src:
            if file_idx in test_file_idx:
                test_mfc_src.append(mfc_src)
                test_mfc_tar.append(mfc_tar)
                test_f0_src.append(f0_src)
                test_f0_tar.append(f0_tar)
                test_ec_src.append(ec_src)
                test_ec_tar.append(ec_tar)
                test_files.append(int(os.path.basename(s)[:-4]))
            else:
                if np.random.rand()<args.eval_size:
                    valid_mfc_src.append(mfc_src)
                    valid_mfc_tar.append(mfc_tar)
                    valid_f0_src.append(f0_src)
                    valid_f0_tar.append(f0_tar)
                    valid_ec_src.append(ec_src)
                    valid_ec_tar.append(ec_tar)
                    valid_files.append(int(os.path.basename(s)[:-4]))
                else:
                    train_mfc_src.append(mfc_src)
                    train_mfc_tar.append(mfc_tar)
                    train_f0_src.append(f0_src)
                    train_f0_tar.append(f0_tar)
                    train_ec_src.append(ec_src)
                    train_ec_tar.append(ec_tar)
                    train_files.append(int(os.path.basename(s)[:-4]))
    
    data_dict = {
            'train_mfc_feat_src':np.asarray(train_mfc_src, np.float32), 
            'train_mfc_feat_tar':np.asarray(train_mfc_tar, np.float32),
            'train_f0_feat_src':np.asarray(train_f0_src, np.float32),
            'train_f0_feat_tar':np.asarray(train_f0_tar, np.float32),
            'train_ec_feat_src':np.asarray(train_ec_src, np.float32),
            'train_ec_feat_tar':np.asarray(train_ec_tar, np.float32),
            'valid_mfc_feat_src':np.asarray(valid_mfc_src, np.float32), 
            'valid_mfc_feat_tar':np.asarray(valid_mfc_tar, np.float32),
            'valid_f0_feat_src':np.asarray(valid_f0_src, np.float32),
            'valid_f0_feat_tar':np.asarray(valid_f0_tar, np.float32),
            'valid_ec_feat_src':np.asarray(valid_ec_src, np.float32),
            'valid_ec_feat_tar':np.asarray(valid_ec_tar, np.float32),
            'test_mfc_feat_src':np.asarray(test_mfc_src, np.float32), 
            'test_mfc_feat_tar':np.asarray(test_mfc_tar, np.float32),
            'test_f0_feat_src':np.asarray(test_f0_src, np.float32),
            'test_f0_feat_tar':np.asarray(test_f0_tar, np.float32),
            'test_ec_feat_src':np.asarray(test_ec_src, np.float32),
            'test_ec_feat_tar':np.asarray(test_ec_tar, np.float32),
            'train_files':np.reshape(np.array(train_files), (-1,1)), 
            'valid_files':np.reshape(np.array(valid_files), (-1,1)), 
            'test_files':np.reshape(np.array(test_files), (-1,1))
            }
    return data_dict


if __name__ == '__main__':
    hparams = Hparams()
    parser = hparams.parser
    hp = parser.parse_args()
    hp.data_folder = '/home/ravi/Downloads/Emo-Conv/neutral-sad/all_above_0.5'
    hp.emo_pair = 'neutral_sad'
    for i in range(1, 6):
        hp.fold = i
        data_dict = extract_features(hp)
        scio.savemat('/home/ravi/Desktop/neu-sad_fold_{0}.mat'.format(i), data_dict)
        del data_dict