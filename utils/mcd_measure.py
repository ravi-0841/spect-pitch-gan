#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 25 16:59:39 2021

@author: ravi
"""

import librosa
import scipy.io.wavfile as scwav
import numpy as np
import os

from glob import glob
from feat_utils import normalize_wav
from pesq import pesq

def mcd(C, C_hat):
    """C and C_hat are NumPy arrays of shape (T, D),
    representing mel-cepstral coefficients.

    """
    K = 10 / np.log(10) * np.sqrt(2)
    return K * np.mean(np.sqrt(np.sum((C - C_hat) ** 2, axis=1)))


#file_dir = '/home/ravi/Desktop/F0_sum_ec/neu-sad/test'
file_dir = '/home/ravi/Desktop/Interspeech-2020/pitch-lddmm-spect/test_crowd_sourcing/vesus/neu-sad/cycle_gan'
fs = sorted(glob(os.path.join(file_dir, '*_target.wav')))

mcd_score = []
pesq_nb = []
pesq_wb = []

hop_len = 0.01
win_len = 0.025

for f in fs:
    try:
        
        basename = os.path.basename(f)
        fid = basename.split('_')[0]
        
        fs, gen = scwav.read(os.path.join(file_dir, '{}.wav'.format(fid)))
        fs, tar = scwav.read(f)
        
        gen = np.asarray(gen, np.float64)
        tar = np.asarray(tar, np.float64)
        gen = normalize_wav(gen)
        tar = normalize_wav(tar)

        pesq_wb.append(pesq(fs, tar, gen, 'wb'))
        pesq_nb.append(pesq(fs, tar, gen, 'nb'))
        
        pesq_nb = [x for x in pesq_nb if np.isnan(x) == False]
        pesq_wb = [x for x in pesq_wb if np.isnan(x) == False]

#        
#        gen_mfcc = librosa.feature.mfcc(y=gen, sr=fs, hop_length=int(fs*hop_len), 
#                                        win_length=int(fs*win_len), 
#                                        n_fft=1024, n_mels=128)
#            
#        tar_mfcc = librosa.feature.mfcc(y=tar, sr=fs, hop_length=int(fs*hop_len), 
#                                        win_length=int(fs*win_len), 
#                                        n_fft=1024, n_mels=128)
#    
#        _, cords = librosa.sequence.dtw(X=gen_mfcc, Y=tar_mfcc, metric='cosine')
#        
#        ext_gen_mfcc = list()
#        ext_tar_mfcc = list()
#        
#        gen_mfcc = gen_mfcc.T
#        tar_mfcc = tar_mfcc.T
#        
#        for i in range(len(cords)-1, -1, -1):
#            ext_gen_mfcc.append(gen_mfcc[cords[i,0],:])
#            ext_tar_mfcc.append(tar_mfcc[cords[i,1],:])
#    
#        ext_gen_mfcc = np.asarray(ext_gen_mfcc)
#        ext_tar_mfcc = np.asarray(ext_tar_mfcc)
#        mcd_score.append(mcd(ext_tar_mfcc, ext_gen_mfcc))

    except Exception as ex:
        print(ex)

    
    
    
    
    