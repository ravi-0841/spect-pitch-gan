#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 18 14:04:08 2021

@author: ravi
"""

import argparse
import os
import numpy as np
import scipy.io.wavfile as scwav
import pylab
import scipy.signal as scisig

import utils.preprocess as preproc
from collections import defaultdict

from utils.feat_utils import preprocess_contour, normalize_wav
from nn_models.model_energy_f0_momenta_wasserstein import VariationalCycleGAN as VCGAN

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

model_dictionary = defaultdict(dict)
model_dictionary['neu-ang'][1] = '/home/ravi/Desktop/spect-pitch-gan/model/neu-ang/folds/lp_1e-05_le_0.1_li_0.0_neu-ang_fold_1_run_4_random_seed_21_epoch_200/neu-ang_200.ckpt'
model_dictionary['neu-ang'][2] = '/home/ravi/Desktop/spect-pitch-gan/model/neu-ang/folds/lp_1e-05_le_0.1_li_0.0_neu-ang_fold_2_run_3_random_seed_21_epoch_200/neu-ang_200.ckpt'
model_dictionary['neu-ang'][3] = '/home/ravi/Desktop/spect-pitch-gan/model/neu-ang/folds/lp_1e-05_le_0.1_li_0.0_neu-ang_fold_3_run_2_random_seed_21_epoch_200/neu-ang_200.ckpt'
model_dictionary['neu-ang'][4] = '/home/ravi/Desktop/spect-pitch-gan/model/neu-ang/folds/lp_1e-05_le_0.1_li_0.0_neu-ang_fold_4_run_1_random_seed_21_epoch_300/neu-ang_300.ckpt'
model_dictionary['neu-ang'][5] = '/home/ravi/Desktop/spect-pitch-gan/model/neu-ang/folds/lp_1e-05_le_0.1_li_0.0_neu-ang_fold_5_run_2_random_seed_21_epoch_300/neu-ang_300.ckpt'

#model_dictionary['neu-ang'][1] = '/home/ravi/Desktop/model_spect_pitch_gan/neu-ang/sum_mfc_wstn_neu-ang_fold_1/lp_1e-05_le_0.1_li_0.0_neu-ang_fold_1_run_4_random_seed_21/neu-ang_100.ckpt'
#model_dictionary['neu-ang'][2] = '/home/ravi/Desktop/model_spect_pitch_gan/neu-ang/sum_mfc_wstn_neu-ang_fold_2/lp_1e-05_le_0.1_li_0.0_neu-ang_fold_2_run_3_random_seed_21/neu-ang_100.ckpt'
#model_dictionary['neu-ang'][3] = '/home/ravi/Desktop/model_spect_pitch_gan/neu-ang/sum_mfc_wstn_neu-ang_fold_3/lp_1e-05_le_0.1_li_0.0_neu-ang_fold_3_run_3_random_seed_21/neu-ang_100.ckpt'
#model_dictionary['neu-ang'][4] = '/home/ravi/Desktop/model_spect_pitch_gan/neu-ang/sum_mfc_wstn_neu-ang_fold_4/lp_1e-05_le_0.1_li_0.0_neu-ang_fold_4_run_4_random_seed_21/neu-ang_100.ckpt'
#model_dictionary['neu-ang'][5] = '/home/ravi/Desktop/model_spect_pitch_gan/neu-ang/sum_mfc_wstn_neu-ang_fold_5/lp_1e-05_le_0.1_li_0.0_neu-ang_fold_5_run_3_random_seed_21/neu-ang_300.ckpt'

model_dictionary['neu-hap'][1] = '/home/ravi/Desktop/spect-pitch-gan/model/neu-hap/folds/lp_0.0001_le_0.001_li_0.0_neu-hap_fold_1_run_3_random_seed_4_epoch_200/neu-hap_200.ckpt'
model_dictionary['neu-hap'][2] = '/home/ravi/Desktop/spect-pitch-gan/model/neu-hap/folds/lp_0.0001_le_0.001_li_0.0_neu-hap_fold_2_run_4_random_seed_4_epoch_300/neu-hap_300.ckpt'
model_dictionary['neu-hap'][3] = '/home/ravi/Desktop/spect-pitch-gan/model/neu-hap/folds/lp_0.0001_le_0.001_li_0.0_neu-hap_fold_3_run_2_random_seed_4_epoch_300/neu-hap_300.ckpt'
model_dictionary['neu-hap'][4] = '/home/ravi/Desktop/spect-pitch-gan/model/neu-hap/folds/lp_0.0001_le_0.001_li_0.0_neu-hap_fold_4_run_2_random_seed_4_epoch_300/neu-hap_300.ckpt'
model_dictionary['neu-hap'][5] = '/home/ravi/Desktop/spect-pitch-gan/model/neu-hap/folds/lp_0.0001_le_0.001_li_0.0_neu-hap_fold_5_run_2_random_seed_4_epoch_400/neu-hap_400.ckpt'

#model_dictionary['neu-hap'][1] = '/home/ravi/Desktop/model_spect_pitch_gan/neu-hap/sum_mfc_wstn_neu-hap_fold_1/lp_0.0001_le_0.001_li_0.0_neu-hap_fold_1_run_3_random_seed_4/neu-hap_100.ckpt'
#model_dictionary['neu-hap'][2] = '/home/ravi/Desktop/model_spect_pitch_gan/neu-hap/sum_mfc_wstn_neu-hap_fold_2/lp_0.0001_le_0.001_li_0.0_neu-hap_fold_2_run_4_random_seed_4/neu-hap_100.ckpt'
#model_dictionary['neu-hap'][3] = '/home/ravi/Desktop/model_spect_pitch_gan/neu-hap/sum_mfc_wstn_neu-hap_fold_3/lp_0.0001_le_0.001_li_0.0_neu-hap_fold_3_run_1_random_seed_4/neu-hap_100.ckpt'
#model_dictionary['neu-hap'][4] = '/home/ravi/Desktop/model_spect_pitch_gan/neu-hap/sum_mfc_wstn_neu-hap_fold_4/lp_0.0001_le_0.001_li_0.0_neu-hap_fold_4_run_3_random_seed_4/neu-hap_100.ckpt'
#model_dictionary['neu-hap'][5] = '/home/ravi/Desktop/model_spect_pitch_gan/neu-hap/sum_mfc_wstn_neu-hap_fold_5/lp_0.0001_le_0.001_li_0.0_neu-hap_fold_5_run_2_random_seed_4/neu-hap_300.ckpt'

model_dictionary['neu-sad'][1] = '/home/ravi/Desktop/spect-pitch-gan/model/neu-sad/folds/lp_0.0001_le_0.1_li_0.0_neu-sad_fold_1_run_4_random_seed_11_epoch_100/neu-sad_100.ckpt'
model_dictionary['neu-sad'][2] = '/home/ravi/Desktop/spect-pitch-gan/model/neu-sad/folds/lp_0.0001_le_0.1_li_0.0_neu-sad_fold_2_run_4_random_seed_11_epoch_100/neu-sad_100.ckpt'
model_dictionary['neu-sad'][3] = '/home/ravi/Desktop/spect-pitch-gan/model/neu-sad/folds/lp_0.0001_le_0.1_li_0.0_neu-sad_fold_3_run_4_random_seed_11_epoch_100/neu-sad_100.ckpt'
model_dictionary['neu-sad'][4] = '/home/ravi/Desktop/spect-pitch-gan/model/neu-sad/folds/lp_0.0001_le_0.1_li_0.0_neu-sad_fold_4_run_2_random_seed_11_epoch_100/neu-sad_100.ckpt'
model_dictionary['neu-sad'][5] = '/home/ravi/Desktop/spect-pitch-gan/model/neu-sad/folds/lp_0.0001_le_0.1_li_0.0_neu-sad_fold_5_run_1_random_seed_11_epoch_200/neu-sad_200.ckpt'

#model_dictionary['neu-sad'][1] = '/home/ravi/Desktop/model_spect_pitch_gan/neu-sad/sum_mfc_wstn_neu-sad_fold_1/lp_0.0001_le_0.1_li_0.0_neu-sad_fold_1_run_1_random_seed_11/neu-sad_100.ckpt'
#model_dictionary['neu-sad'][2] = '/home/ravi/Desktop/model_spect_pitch_gan/neu-sad/sum_mfc_wstn_neu-sad_fold_2/lp_0.0001_le_0.1_li_0.0_neu-sad_fold_2_run_2_random_seed_11/neu-sad_400.ckpt'
#model_dictionary['neu-sad'][3] = '/home/ravi/Desktop/model_spect_pitch_gan/neu-sad/sum_mfc_wstn_neu-sad_fold_3/lp_0.0001_le_0.1_li_0.0_neu-sad_fold_3_run_4_random_seed_11/neu-sad_200.ckpt'
#model_dictionary['neu-sad'][4] = '/home/ravi/Desktop/model_spect_pitch_gan/neu-sad/sum_mfc_wstn_neu-sad_fold_4/lp_0.0001_le_0.1_li_0.0_neu-sad_fold_4_run_3_random_seed_11/neu-sad_200.ckpt'
#model_dictionary['neu-sad'][5] = '/home/ravi/Desktop/model_spect_pitch_gan/neu-sad/sum_mfc_wstn_neu-sad_fold_5/lp_0.0001_le_0.1_li_0.0_neu-sad_fold_5_run_2_random_seed_11/neu-sad_300.ckpt'


num_mfcc = 23
num_pitch = 1
num_energy = 1
sampling_rate = 16000
frame_period = 5.0


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


def conversion(emo_pair='neu-ang', fold=1, data_dir=None, 
               conversion_direction=None, output_dir=None, 
               only_energy=False):

    model = VCGAN(dim_mfc=23, dim_pitch=1, dim_energy=1, mode='test')
    model.load(filepath=model_dictionary[emo_pair][fold])

    os.makedirs(output_dir, exist_ok=True)

    for file in os.listdir(data_dir):

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

        f0_converted, f0_momenta, ec_converted, ec_momenta = model.test(input_pitch=f0, 
                                                      input_mfc=coded_sp,
                                                      input_energy=ec,
                                                      direction=conversion_direction)
        
        ec_converted = np.reshape(ec_converted, (-1,))
        ec_z_idx = np.where(ec_converted>0)[0]
        ec_converted[ec_z_idx] = -1e-6
        
#        pylab.figure(figsize=(13,10))
#        pylab.subplot(311)
#        pylab.plot(ec.reshape(-1,), label='Energy')
#        pylab.plot(ec_converted.reshape(-1,), label='Converted energy')
#        pylab.plot(ec_momenta.reshape(-1,), label='Energy momenta')
#        pylab.legend()
#        pylab.subplot(312)
#        pylab.plot(f0.reshape(-1,), label='F0')
#        pylab.plot(f0_converted.reshape(-1,), label='Converted F0')
#        pylab.plot(f0_momenta.reshape(-1,), label='F0 momenta')
#        pylab.legend()
#        pylab.subplot(313)
#        pylab.plot(np.divide(ec_converted.reshape(-1,), ec.reshape(-1,)), label='Energy Ratio')
#        pylab.legend()
#        pylab.savefig(os.path.join(output_dir, os.path.basename(filepath)[:-4])+'.png')
#        pylab.close()
        
        f0_converted = np.asarray(np.reshape(f0_converted[0], (-1,)), np.float64)
        f0_converted = np.ascontiguousarray(f0_converted)
        f0_converted[f0_z_idx] = 0
        
        # Modifying the spectrum instead of mfcc
        decoded_sp_converted = np.multiply(sp.T, np.divide(ec_converted.reshape(1,-1), 
                                    ec.reshape(1,-1)))
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


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description = 'Convert Emotion using VariationalCycleGAN model.')

    emo_pair_dict = {
            'neu-ang': 'neutral-angry', 
            'neu-hap': 'neutral-happy', 
            'neu-sad': 'neutral-sad'
            }

    emo_pair_default = 'neu-ang'
    fold_default = 3
    conversion_direction_default = 'A2B'

    parser.add_argument('--emo_pair', type=str, help='Emotion pair', default=emo_pair_default)
    parser.add_argument('--fold', type=int, help='Fold', default=fold_default)
    parser.add_argument('--conversion_direction', type=str, help='Conversion direction for VCGAN, A2B or B2A', 
                        default=conversion_direction_default)

    argv = parser.parse_args()

    emo_pair = argv.emo_pair
    fold = argv.fold
    conversion_direction = argv.conversion_direction
    data_dir = '/home/ravi/Downloads/Emo-Conv/{}/speaker_folds/paired_folds/fold{}/test/neutral'.format(emo_pair_dict[emo_pair], fold)
    output_dir = '/home/ravi/Desktop/F0_sum_ec_new/second_run/{}/fold{}'.format(emo_pair, fold)
    
    conversion(emo_pair=emo_pair, fold=fold, data_dir=data_dir, 
               conversion_direction=conversion_direction, 
               output_dir=output_dir, only_energy=True)


