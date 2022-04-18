#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  5 10:17:03 2022

@author: ravi
"""

import argparse
import os
import numpy as np
import scipy.io.wavfile as scwav
import pylab
import scipy.signal as scisig

import utils.preprocess as preproc
from glob import glob
from utils.feat_utils import preprocess_contour, normalize_wav
from nn_models.model_energy_f0_momenta_wasserstein import VariationalCycleGAN as VCGAN


num_mfcc = 23
num_pitch = 1
num_energy = 1
sampling_rate = 16000
frame_period = 5.0


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

model_dir_dict = {
        'neu-ang': '/home/ravi/Desktop/spect-pitch-gan/model/neu-ang/sum_mfc_wstn_spk5_neu-ang/lp_1e-05_le_0.1_li_0.0_lrg_1e-05_lrd_1e-07_sum_mfc_spk5_neu-ang',
        'neu-hap': '/home/ravi/Desktop/spect-pitch-gan/model/neu-hap/sum_mfc_wstn_spk5_neu-hap/lp_0.0001_le_0.001_li_0.0_lrg_1e-05_lrd_1e-07_sum_mfc_spk5_neu-hap',
        'neu-sad': '/home/ravi/Desktop/spect-pitch-gan/model/neu-sad/sum_mfc_wstn_spk5_neu-sad/lp_0.0001_le_0.1_li_0.0_lrg_1e-05_lrd_1e-07_sum_mfc_spk5_neu-sad'
        }

model_file_dict = {
        'neu-ang': 'neu-ang_800.ckpt',
        'neu-hap': 'neu-hap_1000.ckpt',
        'neu-sad': 'neu-sad_1000.ckpt'
        }



def conversion(model_dir=None, model_name=None, data_dir=None, 
               conversion_direction=None, output_dir=None, 
               embedding=True, only_energy=False):

    model = VCGAN(dim_mfc=23, dim_pitch=1, dim_energy=1, mode='test')
    model.load(filepath=os.path.join(model_dir, model_name))

    os.makedirs(output_dir, exist_ok=True)
    
    files = sorted(glob(os.path.join(data_dir, '*.wav')))
#        for file in os.listdir(data_dir):
    for file in files:
        try:
            filepath = os.path.join(data_dir, file)
            
            sr, wav = scwav.read(filepath)
            wav = np.asarray(wav, np.float64)
            wav = normalize_wav(wav, floor=-1, ceil=1)
            assert (sr==sampling_rate)
            
            wav = preproc.wav_padding(wav=wav, sr=sampling_rate, \
                                  frame_period=frame_period, multiple=4)
                
            f0, sp, ap = preproc.world_decompose(wav=wav, \
                            fs=sampling_rate, frame_period=frame_period)
            
            coded_sp = preproc.world_encode_spectral_envelope(sp=sp, \
                                fs=sampling_rate, dim=num_mfcc)
            ec = np.reshape(np.sum(coded_sp, axis=-1), (-1,)) + 1e-06
            
            coded_sp = np.expand_dims(coded_sp, axis=0)
            coded_sp = np.transpose(coded_sp, (0,2,1))
            
            f0_z_idx = np.where(f0<10.0)[0]
            ec_z_idx = np.where(ec>0)[0]
            ec[ec_z_idx] = -1e-6
    
            f0 = preprocess_contour(f0)
            ec = preprocess_contour(ec)
    
            f0 = np.reshape(f0, (1,1,-1))
            ec = np.reshape(ec, (1,1,-1))
    
            f0_converted, f0_momenta, \
                ec_converted, ec_momenta = model.test(input_pitch=f0, 
                                                      input_mfc=coded_sp,
                                                      input_energy=ec,
                                                      direction=conversion_direction)
            
            ec_converted = np.reshape(ec_converted, (-1,))
            ec_z_idx = np.where(ec_converted>0)[0]
            ec_converted[ec_z_idx] = -1e-6

            f0_converted = np.asarray(np.reshape(f0_converted[0], (-1,)), np.float64)
            f0_converted = np.ascontiguousarray(f0_converted)
            f0_converted[f0_z_idx] = 0
            
            # Modifying the spectrum instead of mfcc
            decoded_sp_converted = np.multiply(sp.T, np.divide(ec_converted.reshape(1,-1), 
                                        ec.reshape(1,-1)))
            # decoded_sp_converted = np.multiply(sp.T, 1.)
            decoded_sp_converted = np.ascontiguousarray(decoded_sp_converted.T)
            
            decoded_sp_converted = decoded_sp_converted[10:-10]
            f0_converted = f0_converted[10:-10]
            ap = ap[10:-10]
            
            wav_transformed = preproc.world_speech_synthesis(f0=f0_converted, 
                                                             decoded_sp=decoded_sp_converted, 
                                                             ap=ap, fs=sampling_rate, 
                                                             frame_period=frame_period)
            
            wav_transformed = -1 + 2*(wav_transformed - np.min(wav_transformed)) \
                    / (np.max(wav_transformed) - np.min(wav_transformed))
            wav_transformed = wav_transformed - np.mean(wav_transformed)
            
            scwav.write(os.path.join(output_dir, os.path.basename(filepath)), 
                        16000, wav_transformed)
            print('Processed: ' + filepath)
        except Exception as ex:
            print(ex)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description = 'Convert Emotion using VariationalCycleGAN model.')

    target_dict = {'neu-ang':'angry', 'neu-hap':'happy', 'neu-sad':'sad'}

    emo_pair_default = 'neu-ang'
    data_dir_default = '/home/ravi/Downloads/Emo-Conv/neutral-{}/test_5_7/neutral'.format(target_dict[emo_pair_default])
    conversion_direction_default = 'A2B'
    # output_dir_default = '/home/ravi/Desktop/F0_sum_ec/F0/{}'.format(emo_pair_default)

    parser.add_argument('--emo_pair', type=str, help='Emotion pair.', default=emo_pair_default)
    parser.add_argument('--data_dir', type=str, help='Directory for the voices for conversion.', default=data_dir_default)
    parser.add_argument('--conversion_direction', type=str, help='Conversion direction for VCGAN, A2B or B2A', default=conversion_direction_default)

    argv = parser.parse_args()

    emo_pair = argv.emo_pair
    data_dir = argv.data_dir
    conversion_direction = argv.conversion_direction

    model_dir = model_dir_dict[emo_pair]
    model_name = model_file_dict[emo_pair]
    output_dir = '/home/ravi/Desktop/VCGAN/spk5/F0+Ec/{}'.format(emo_pair)
    
    conversion(model_dir=model_dir, model_name=model_name, data_dir=data_dir, 
               conversion_direction=conversion_direction, output_dir=output_dir, 
               embedding=True, only_energy=True)
