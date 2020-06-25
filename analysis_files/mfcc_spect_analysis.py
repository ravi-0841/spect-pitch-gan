#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 12:23:34 2020

@author: ravi
"""

import librosa
import numpy as np
import pyworld as pw
import scipy.io.wavfile as scwav
import pylab
import scipy
from scipy.optimize import nnls, fmin_l_bfgs_b
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from functools import partial


def _power_to_db(S):
    return 20*np.log10(S)


def _db_to_power(S):
    return 10**(S / 20)


def _nnls(A,b):
    return nnls(A,b)


def _smooth(x, window_len=7, window='hanning'):
    if x.ndim != 1:
        raise(ValueError, "smooth only accepts 1 dimension arrays.")
    if x.size < window_len:
        print(len(x))
        raise(ValueError, "Input vector needs to be bigger than window size.")
    if window_len<3:
        return x
    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise(ValueError, "Window is out of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")
    s = np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
    #print(len(s))
    if window == 'flat': #moving average
        w = np.ones(window_len,'d')
    else:
        w = eval('np.'+window+'(window_len)')
    y = np.convolve(w/w.sum(),s,mode='same')
    return y[window_len-1:-1*(window_len-1)]


def _nnls_obj(x, shape, A, B):
    '''Compute the objective and gradient for NNLS'''

    # Scipy's lbfgs flattens all arrays, so we first reshape
    # the iterate x
    x = x.reshape(shape)

    # Compute the difference matrix
    diff = np.dot(A, x) - B

    # Compute the objective value
    value = 0.5 * np.sum(diff**2)

    # And the gradient
    grad = np.dot(A.T, diff)

    # Flatten the gradient
    return value, grad.flatten()


def smooth_spectrum(A, window_len=27, window='flat'):
    time_steps = A.shape[0]
    smoothed_spectrum = np.empty((0, 513))
    executor = ProcessPoolExecutor(max_workers=8)
    futures = []

    for i in range(time_steps):
        futures.append(executor.submit(partial(_smooth, A[i,:], 
                                               window_len=window_len, 
                                               window=window)))
    
    results = [future.result() for future in futures] #tqdm(futures)
    
    for result in results:
        smoothed_spectrum = np.concatenate((smoothed_spectrum, 
                                            np.reshape(result, (1, 513))), axis=0)
    return smoothed_spectrum


def nnls_lbfgs_block(A, B, x_init=None, **kwargs):
    '''Solve the constrained problem over a single block
    Parameters
    ----------
    A : np.ndarray [shape=(m, d)]
        The basis matrix
    B : np.ndarray [shape=(m, N)]
        The regression targets
    x_init : np.ndarray [shape=(d, N)]
        An initial guess
    kwargs
        Additional keyword arguments to `scipy.optimize.fmin_l_bfgs_b`
    Returns
    -------
    x : np.ndarray [shape=(d, N)]
        Non-negative matrix such that Ax ~= B
    '''

    # If we don't have an initial point, start at the projected
    # least squares solution
    if x_init is None:
        x_init = np.linalg.lstsq(A, B, rcond=None)[0]
        np.clip(x_init, 0, None, out=x_init)

    # Adapt the hessian approximation to the dimension of the problem
    kwargs.setdefault('m', A.shape[1])

    # Construct non-negative bounds
    bounds = [(0, None)] * x_init.size
    shape = x_init.shape

    # optimize
    x, obj_value, diagnostics = fmin_l_bfgs_b(_nnls_obj, x_init,
                                              args=(shape, A, B),
                                              bounds=bounds,
                                              **kwargs)
    # reshape the solution
    return x.reshape(shape)


def compute_power_spectrum_from_mel(A, B):
    executor = ProcessPoolExecutor(max_workers=8)
    futures = []

    for i in range(B.shape[-1]):
        futures.append(executor.submit(partial(_nnls, A, B[:,i])))
    
    results = [future.result() for future in futures] # tqdm(futures)
    
    spectrum = np.empty((513,0))
    for result in results:
        spectrum = np.concatenate((spectrum, np.reshape(result[0], (513,1))), 
                                  axis=1)
    return spectrum


N_FFT = 1024
HOP_LENGTH = 80
WIN_LENGTH = 80
NUM_MFCC = 23
NUM_MELS = 128
POWER_EXP = 2.0
FRAME_PERIOD = 5



if __name__=='__main__':
    sr, data = scwav.read('./spect-pitch-gan/data/CMU-ARCTIC-US/cmu_us_f1/wav/arctic_a0004.wav')
    data = np.asarray(data, np.float64)
    data = (data - np.min(data)) / (np.max(data) - np.min(data))
    data = data - np.mean(data)

#    stft = np.abs(librosa.stft(data, n_fft=N_FFT, hop_length=HOP_LENGTH, 
#                               win_length=WIN_LENGTH, pad_mode='constant'))
#    filters = librosa.filters.mel(sr, n_fft=N_FFT, n_mels=NUM_MELS, htk=True)
#    mel_spec = librosa.power_to_db(np.dot(filters, stft**2))
#    pylab.figure(), pylab.imshow(np.asarray(mel_spec, np.float64))
#    pylab.title('Mel spectrogram')
    
#    mfcc = scipy.fftpack.dct(mel_spec, axis=0, type=2, norm='ortho')
#    pylab.figure(), pylab.plot(mfcc[0,:])
#    pylab.title('MFCC features')

    """
    Pyworld Analysis
    """
    
    f0, world_spect, aperiod = pw.wav2world(data, sr, frame_period=FRAME_PERIOD)
    world_mfcc = pw.code_spectral_envelope(world_spect, sr, NUM_MFCC)
    fftlen = pw.get_cheaptrick_fft_size(sr)
    world_spect_back = pw.decode_spectral_envelope(world_mfcc, sr, fftlen)

#    pylab.figure(), pylab.imshow(librosa.power_to_db(world_spect.T ** 2))
#    pylab.title('World Spectrogram')
    
#    pylab.figure(), pylab.imshow(librosa.power_to_db(world_spect_back.T ** 2))
#    pylab.title('World Spectrogram FB')
    
    filters = librosa.filters.mel(sr, n_fft=N_FFT, n_mels=NUM_MELS, htk=True)
    world_spect_mel = np.dot(world_spect**POWER_EXP, filters.T)
    world_spect_mel_power = _power_to_db(world_spect_mel)
    world_spect_mel_mfcc = scipy.fftpack.dct(world_spect_mel_power, 
                                             axis=1, type=2, norm='ortho')[:,:NUM_MFCC]
#    pylab.figure(), pylab.subplot(211), pylab.imshow(world_mfcc.T), pylab.title('World original MFCC')
#    pylab.subplot(212), pylab.imshow(world_spect_mel_mfcc.T), pylab.title('World spect MFCC')
    
    
    # Inverting world spect from mfcc to mel to spect
    world_spect_mel_mfcc_mel_power = scipy.fftpack.idct(world_spect_mel_mfcc, axis=1, 
                                                  type=2, norm='ortho', n=NUM_MELS)
    
#    pylab.figure(), pylab.subplot(211), pylab.imshow(world_spect_mel_power.T)
#    pylab.title('World original mel')
#    pylab.subplot(212), pylab.imshow(world_spect_mel_mfcc_mel_power.T)
#    pylab.title('World recon mel')
    
    world_spect_mel_mfcc_mel = _db_to_power(world_spect_mel_mfcc_mel_power)
    world_spect_mel_mfcc_mel_power = compute_power_spectrum_from_mel(filters, 
                                                    world_spect_mel_mfcc_mel.T)
    world_spect_mel_mfcc_mel_spect = np.transpose(np.power(world_spect_mel_mfcc_mel_power, 1/POWER_EXP))
    world_spect_mel_mfcc_mel_spect_librosa = np.transpose(np.power(nnls_lbfgs_block(np.asarray(filters, np.float64), 
                                                   world_spect_mel_mfcc_mel.T), 1/POWER_EXP))

#    pylab.figure(), pylab.subplot(221), pylab.imshow(world_spect.T), pylab.title('World original spect')
#    pylab.subplot(222), pylab.imshow(world_spect_mel_mfcc_mel_spect.T), pylab.title('World recon spect')
#    pylab.subplot(223), pylab.imshow(world_spect_mel_mfcc_mel_spect_librosa.T), pylab.title('World recon spect (librosa)')
#    pylab.subplot(224), pylab.imshow(world_spect_mel_mfcc_mel_spect_librosa.T), pylab.title('World recon spect (librosa)')
    
    world_spect_mel_mfcc_mel_spect_librosa = np.ascontiguousarray(world_spect_mel_mfcc_mel_spect_librosa)
#    world_spect_mel_mfcc_mel_spect_librosa = (world_spect_mel_mfcc_mel_spect_librosa \
#                                              - np.min(world_spect_mel_mfcc_mel_spect_librosa)) \
#                                              / (np.max(world_spect_mel_mfcc_mel_spect_librosa) \
#                                                 - np.min(world_spect_mel_mfcc_mel_spect_librosa))
    
    world_spect_mel_mfcc_mel_spect_librosa = smooth_spectrum(world_spect_mel_mfcc_mel_spect_librosa)
    world_spect_mel_mfcc_mel_spect_librosa = world_spect_mel_mfcc_mel_spect_librosa
    speech_recon = pw.synthesize(f0, world_spect_mel_mfcc_mel_spect_librosa, aperiod, sr, frame_period=FRAME_PERIOD)
    speech_original = pw.synthesize(f0, world_spect_back, aperiod, sr, frame_period=FRAME_PERIOD)
    scwav.write('./test_original.wav', sr, np.asarray(speech_original, np.float32))
    scwav.write('./test_recon.wav', sr, np.asarray(speech_recon, np.float32))
    
    pylab.figure()
    q = np.random.randint(0, world_spect_mel_mfcc_mel_spect_librosa.shape[0])
    pylab.plot(world_spect_back[q,:]/np.max(world_spect_back), 'g', label='original')
    pylab.plot(world_spect_mel_mfcc_mel_spect_librosa[q,:]/np.max(world_spect_mel_mfcc_mel_spect_librosa), 
               'r', label='recon')
    pylab.legend(loc=1)
    pylab.title('Visualizing %dth frame' % q)
    
    print('nans are:- %f' % np.count_nonzero(np.isnan(speech_recon)))



