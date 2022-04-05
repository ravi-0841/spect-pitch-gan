#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  5 11:27:42 2022

@author: ravi
"""

import argparse
import os
import numpy as np
import tensorflow as tf
import librosa
import soundfile as sf

from glob import glob
from cyclegan_model_mceps import CycleGAN as CycleGAN_mceps

from baseline_CycleGAN.preprocess import *
from baseline_CycleGAN.utils import get_lf0_cwt_norm,norm_scale,denormalize
from baseline_CycleGAN.utils import get_cont_lf0, get_lf0_cwt,inverse_cwt
from sklearn import preprocessing

import utils.preprocess as preproc
from collections import defaultdict

from utils.feat_utils import preprocess_contour, normalize_wav
from nn_models.model_energy_f0_momenta_wasserstein import VariationalCycleGAN as VCGAN

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

model_dictionary = defaultdict(dict)
model_dictionary['neu-ang'] = '/home/ravi/Desktop/spect-pitch-gan/model/neu-ang/sum_mfc_wstn_spk5_neu-ang/lp_1e-05_le_0.1_li_0.0_lrg_1e-05_lrd_1e-07_sum_mfc_spk5_neu-ang/neu-ang_800.ckpt'
model_dictionary['neu-hap'] = '/home/ravi/Desktop/spect-pitch-gan/model/neu-hap/sum_mfc_wstn_spk5_neu-hap/lp_0.0001_le_0.001_li_0.0_lrg_1e-05_lrd_1e-07_sum_mfc_spk5_neu-hap/neu-hap_1000.ckpt'
model_dictionary['neu-sad'] = '/home/ravi/Desktop/spect-pitch-gan/model/neu-sad/sum_mfc_wstn_spk5_neu-sad/lp_0.0001_le_0.1_li_0.0_lrg_1e-05_lrd_1e-07_sum_mfc_spk5_neu-sad/neu-sad_1000.ckpt'

num_mceps = 24
sampling_rate = 16000
frame_period = 5.0

def f0_conversion(model_f0, f0, coded_sp, ec, direction='A2B'):
    f0_converted, _, _, _ = model_f0.test(input_pitch=f0, input_mfc=coded_sp,
                                          input_energy=ec, direction=direction)
    return f0_converted


def mcep_conversion(model_mceps, features, direction):
    coded_sp_converted_norm = model_mceps.test(inputs=features, 
                                               direction=conversion_direction)[0]
    return coded_sp_converted_norm


def conversion(model_mceps, model_f0, mcep_mean_A, 
               mcep_std_A, mcep_mean_B, mcep_std_B, 
               data_dir, conversion_direction, output_dir):

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    files = sorted(glob(os.path.join(data_dir, '*.wav')))
    for file in files:
        try:
            wav, _ = librosa.load(file, sr=sampling_rate, mono=True)
            wav = np.asarray(wav, np.float64)
            wav = normalize_wav(wav, floor=-1, ceil=1)
            wav = wav_padding(wav=wav, sr=sampling_rate, 
                              frame_period=frame_period, multiple=4)
            f0, timeaxis, sp, ap = world_decompose(wav=wav, 
                                                   fs=sampling_rate, 
                                                   frame_period=frame_period)
            coded_sp_vcg = world_encode_spectral_envelop(sp=sp, 
                                                     fs=sampling_rate, 
                                                     dim=num_mceps-1)
            
            coded_sp = world_encode_spectral_envelop(sp=sp, 
                                                     fs=sampling_rate, 
                                                     dim=num_mceps)
            
            coded_sp_transposed = coded_sp.T

            ec = np.reshape(np.sum(coded_sp, axis=-1), (-1,)) + 1e-06
            coded_sp_f0_input = np.transpose(np.expand_dims(coded_sp_vcg, axis=0), 
                                             axes=(0,2,1))
            f0_z_idx = np.where(f0<10.0)[0]
            ec_z_idx = np.where(ec>0)[0]
            ec[ec_z_idx] = -1e-6
    
            f0 = preprocess_contour(f0)
            ec = preprocess_contour(ec)
    
            f0 = np.reshape(f0, (1,1,-1))
            ec = np.reshape(ec, (1,1,-1))
            
            f0_converted = f0_conversion(model_f0, f0, coded_sp_f0_input, ec)
            f0_converted = np.asarray(np.reshape(f0_converted[0], (-1,)), np.float64)
            f0_converted = np.ascontiguousarray(f0_converted)
            f0_converted[f0_z_idx] = 0
            
            coded_sp_norm = (coded_sp_transposed - mcep_mean_A) / mcep_std_A

            # test mceps
            coded_sp_converted_norm = mcep_conversion(model_mceps=model_mceps, 
                                                      features=np.array([coded_sp_norm]), 
                                                      direction=conversion_direction)

            coded_sp_converted = coded_sp_converted_norm * mcep_std_B + mcep_mean_B
    
            coded_sp_converted = coded_sp_converted.T
            coded_sp_converted = np.ascontiguousarray(np.asarray(coded_sp_converted, np.float64))
            decoded_sp_converted = world_decode_spectral_envelop(coded_sp=coded_sp_converted, 
                                                                 fs=sampling_rate)
            wav_transformed = world_speech_synthesis(f0=f0_converted, 
                                                     decoded_sp=decoded_sp_converted, 
                                                     ap=ap, fs=sampling_rate, 
                                                     frame_period=frame_period)

            fid = os.path.basename(file)
            sf.write(os.path.join(output_dir, fid), 
                                     wav_transformed, sampling_rate)

            print("Processed "+file)
        except Exception as ex:
            print(ex)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Convert voices using pre-trained CycleGAN model.')
    
    emo_pair_default = 'neutral_to_sad'
    emo_pair_dict = {
            'neutral_to_angry':'neu-ang', 
            'neutral_to_happy':'neu-hap', 
            'neutral_to_sad':'neu-sad'
            }
    emo_pair_dict_data = {
            'neutral_to_angry':'neutral-angry', 
            'neutral_to_happy':'neutral-happy', 
            'neutral_to_sad':'neutral-sad'
            }

    parser.add_argument('--emo_pair', type=str, help='Emotion pair', 
                        default=emo_pair_default, 
                        choices=['neutral_to_angry', 'neutral_to_happy', 'neutral_to_sad'])
    parser.add_argument('--fold', type=int, help='Fold', default=1)

    argv = parser.parse_args()
    
    emo_pair = argv.emo_pair
    target = emo_pair.split('_')[-1]

    model_mceps_dir = '/home/ravi/Desktop/baseline_cycleGAN/model/{}_mceps_spk5/'.format(emo_pair)
    model_mceps_file = os.path.join(model_mceps_dir, 'mceps.ckpt')
    data_dir = '/home/ravi/Downloads/Emo-Conv/neutral-{}/test_5_7/neutral'.format(target)
    conversion_direction = 'A2B'
    output_dir = '/home/ravi/Desktop/VCGAN_CycleGAN/{}'.format(emo_pair_dict[emo_pair])

    graph_mceps = tf.Graph()
    graph_pitch = tf.Graph()
    
    with graph_mceps.as_default():
        model_mceps = CycleGAN_mceps(num_features=num_mceps, mode='test')
        model_mceps.load(filepath=model_mceps_file)    
    print("Mceps model loaded")
    
    with graph_pitch.as_default():
        model_f0 = VCGAN(dim_mfc=num_mceps-1, dim_pitch=1, dim_energy=1, mode='test')
        model_f0.load(filepath=model_dictionary[emo_pair_dict[emo_pair]])
    print("Pitch model loaded")
    
    mcep_normalization_params = np.load(os.path.join(model_mceps_dir, 'mcep_normalization.npz'))
    mcep_mean_A = mcep_normalization_params['mean_A']
    mcep_std_A = mcep_normalization_params['std_A']
    mcep_mean_B = mcep_normalization_params['mean_B']
    mcep_std_B = mcep_normalization_params['std_B']


    conversion(model_mceps, model_f0, mcep_mean_A, mcep_std_A, 
               mcep_mean_B, mcep_std_B, data_dir=data_dir, 
               conversion_direction=conversion_direction, 
               output_dir=output_dir)


