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
        'neu-ang': '/home/ravi/Desktop/crowd-sourcing/all_samples/F0_sum_ec/mixed_and_raw_models/sum_mfc_models/neu-ang/lp_1e-05_le_0.1_li_0.0_lrg_1e-05_lrd_1e-07_sum_mfc_epoch_200_best',
        'neu-hap': '/home/ravi/Desktop/crowd-sourcing/all_samples/F0_sum_ec/mixed_and_raw_models/sum_mfc_models/neu-hap/lp_0.0001_le_0.001_li_0.0_lrg_1e-05_lrd_1e-07_sum_mfc_gender_neu-hap_random_seed_2_4_epoch_200_best',
        'neu-sad': '/home/ravi/Desktop/crowd-sourcing/all_samples/F0_sum_ec/mixed_and_raw_models/sum_mfc_models/neu-sad/lp_0.0001_le_0.1_li_0.0_lrg_1e-05_lrd_1e-07_sum_mfc_neu-sad_epoch_200_best'
        }

model_file_dict = {
        'neu-ang': 'neu-ang_200.ckpt',
        'neu-hap': 'neu-hap_200.ckpt',
        'neu-sad': 'neu-sad_200.ckpt'
        }



def conversion(model_dir=None, model_name=None, audio_file=None, 
               data_dir=None, conversion_direction=None, 
               output_dir=None, embedding=True, only_energy=False):

    model = VCGAN(dim_mfc=23, dim_pitch=1, dim_energy=1, mode='test')
    model.load(filepath=os.path.join(model_dir, model_name))
    
    if audio_file is not None:
        sr, wav = scwav.read(audio_file)
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

        pylab.figure(figsize=(13,10))
        pylab.subplot(311)
        pylab.plot(ec.reshape(-1,), label='Energy')
        pylab.plot(ec_converted.reshape(-1,), label='Converted energy')
        pylab.plot(ec_momenta.reshape(-1,), label='Energy momenta')
        pylab.legend()
        pylab.subplot(312)
        pylab.plot(f0.reshape(-1,), label='F0')
        pylab.plot(f0_converted.reshape(-1,), label='Converted F0')
        pylab.plot(f0_momenta.reshape(-1,), label='F0 momenta')
        pylab.legend()
        pylab.subplot(313)
        pylab.plot(np.divide(ec_converted.reshape(-1,), ec.reshape(-1,)), label='Energy Ratio')
        pylab.legend()

#            coded_sp_converted = np.asarray(np.transpose(np.squeeze(coded_sp_converted)), 
#                                            np.float64)
#            coded_sp_converted = np.ascontiguousarray(coded_sp_converted)
        
        f0_converted = np.asarray(np.reshape(f0_converted[0], (-1,)), np.float64)
        f0_converted = np.ascontiguousarray(f0_converted)
        f0_converted[f0_z_idx] = 0
        
#        coded_sp = np.transpose(np.squeeze(coded_sp))
#        coded_sp_converted = np.multiply(coded_sp.T, np.divide(ec_converted.reshape(1,-1), 
#                                        ec.reshape(1,-1)))
#        coded_sp_converted = np.ascontiguousarray(coded_sp_converted.T)
#            
#        decoded_sp_converted = preproc.world_decode_spectral_envelope(coded_sp=coded_sp_converted, 
#                                                                    fs=sampling_rate)
        
        # Modifying the spectrum instead of mfcc
        decoded_sp_converted = np.multiply(sp.T, np.divide(ec_converted.reshape(1,-1), 
                                    ec.reshape(1,-1)))
        
        decoded_sp_converted = np.ascontiguousarray(decoded_sp_converted.T)
        
        # Normalization of converted features
#        decoded_sp_converted = decoded_sp_converted.T / np.max(decoded_sp_converted)
#        decoded_sp_converted = np.ascontiguousarray(decoded_sp_converted)
#        decoded_sp_converted = decoded_sp_converted[10:-10]
#        f0_converted = f0_converted[10:-10]
#        ap = ap[10:-10]
        
        decoded_sp_converted = decoded_sp_converted[6:-6]
        f0_converted = f0_converted[6:-6]
        ap = ap[6:-6]
        
        wav_transformed = preproc.world_speech_synthesis(f0=f0_converted, 
                                                         decoded_sp=decoded_sp_converted, 
                                                         ap=ap, fs=sampling_rate, 
                                                         frame_period=frame_period)
        
        wav_transformed = -1 + 2*(wav_transformed - np.min(wav_transformed)) \
                / (np.max(wav_transformed) - np.min(wav_transformed))
        wav_transformed = wav_transformed - np.mean(wav_transformed)
        
        scwav.write(os.path.join('/home/ravi/Desktop', 
                                 os.path.basename(audio_file)), 
                                sampling_rate, wav_transformed)
        print('Processed: ' + audio_file)
        
    else:
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
        
                f0_converted, f0_momenta, ec_converted, ec_momenta = model.test(input_pitch=f0, 
                                                              input_mfc=coded_sp,
                                                              input_energy=ec,
                                                              direction=conversion_direction)
                
                ec_converted = np.reshape(ec_converted, (-1,))
                ec_z_idx = np.where(ec_converted>0)[0]
                ec_converted[ec_z_idx] = -1e-6
                
#                pylab.figure(figsize=(13,10))
#                pylab.subplot(311)
#                pylab.plot(ec.reshape(-1,), label='Energy')
#                pylab.plot(ec_converted.reshape(-1,), label='Converted energy')
#                pylab.plot(ec_momenta.reshape(-1,), label='Energy momenta')
#                pylab.legend()
#                pylab.subplot(312)
#                pylab.plot(f0.reshape(-1,), label='F0')
#                pylab.plot(f0_converted.reshape(-1,), label='Converted F0')
#                pylab.plot(f0_momenta.reshape(-1,), label='F0 momenta')
#                pylab.legend()
#                pylab.subplot(313)
#                pylab.plot(np.divide(ec_converted.reshape(-1,), ec.reshape(-1,)), label='Energy Ratio')
#                pylab.legend()
#                pylab.savefig(os.path.join(output_dir, os.path.basename(filepath)[:-4])+'.png')
#                pylab.close()
                
                f0_converted = np.asarray(np.reshape(f0_converted[0], (-1,)), np.float64)
                f0_converted = np.ascontiguousarray(f0_converted)
                f0_converted[f0_z_idx] = 0
                
                # Modifying the spectrum instead of mfcc
#                decoded_sp_converted = np.multiply(sp.T, np.divide(ec_converted.reshape(1,-1), 
#                                            ec.reshape(1,-1)))
                decoded_sp_converted = np.multiply(sp.T, 1.)
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

    emo_pair_default = 'neu-sad'
    data_dir_default = '/home/ravi/Desktop/spect-pitch-gan/data/evaluation/{}/test/neutral'.format(emo_pair_default)
    conversion_direction_default = 'A2B'
    output_dir_default = '/home/ravi/Desktop/F0_sum_ec/onlyF0/{}/vesus'.format(emo_pair_default)
    audio_file_default = None #'/home/ravi/Desktop/spect-pitch-gan/data/evaluation/neu-ang/neutral/418.wav'

    parser.add_argument('--emo_pair', type=str, help='Emotion pair.', default=emo_pair_default)
    parser.add_argument('--data_dir', type=str, help='Directory for the voices for conversion.', default=data_dir_default)
    parser.add_argument('--conversion_direction', type=str, help='Conversion direction for VCGAN, A2B or B2A', default=conversion_direction_default)
    parser.add_argument('--audio_file', type=str, help='convert a single audio file', default=audio_file_default)

    argv = parser.parse_args()

    emo_pair = argv.emo_pair
    data_dir = argv.data_dir
    conversion_direction = argv.conversion_direction
    audio_file = argv.audio_file

    model_dir = model_dir_dict[emo_pair]
    model_name = model_file_dict[emo_pair]
    output_dir = '/home/ravi/Desktop/F0_sum_ec/onlyF0/{}/vesus'.format(emo_pair)
    
    conversion(model_dir=model_dir, model_name=model_name, audio_file=audio_file, 
               data_dir=data_dir, conversion_direction=conversion_direction, 
               output_dir=output_dir, embedding=True, only_energy=True)
