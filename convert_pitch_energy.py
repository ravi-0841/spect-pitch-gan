import argparse
import os
import numpy as np
import scipy.io.wavfile as scwav
import pylab
import scipy.signal as scisig

import utils.preprocess as preproc
from utils.feat_utils import preprocess_contour, normalize_wav
from nn_models.model_energy_f0_momenta_discriminate_wasserstein import VariationalCycleGAN as VCGAN


num_mfcc = 23
num_pitch = 1
num_energy = 1
sampling_rate = 16000
frame_period = 5.0


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


def conversion(model_dir=None, model_name=None, audio_file=None, 
               data_dir=None, conversion_direction=None, 
               output_dir=None, embedding=True, only_energy=False):

    model = VCGAN(dim_mfc=23, dim_pitch=1, dim_energy=1, mode='test')
    model.load(filepath=os.path.join(model_dir, model_name))
    
    if audio_file is not None:
        sr, wav = scwav.read(audio_file)
        wav = np.asarray(wav, np.float64)
        wav = normalize_wav(wav)
        assert (sr==sampling_rate)
        
        wav = preproc.wav_padding(wav=wav, sr=sampling_rate, \
                              frame_period=frame_period, multiple=4)
            
        f0, sp, ap = preproc.world_decompose(wav=wav, \
                        fs=sampling_rate, frame_period=frame_period)
        ec = np.reshape(np.sum(sp**2, axis=1), (-1,))
        pylab.plot(ec, label='original energy')
        
        coded_sp = preproc.world_encode_spectral_envelope(sp=sp, \
                            fs=sampling_rate, dim=num_mfcc)
        
        coded_sp = np.expand_dims(coded_sp, axis=0)
        coded_sp = np.transpose(coded_sp, (0,2,1))
        
        f0_z_idx = np.where(f0<10.0)[0]
        ec_z_idx = np.where(ec<1e-04)[0]

        f0 = preprocess_contour(f0)
        ec = preprocess_contour(ec)

        f0 = np.reshape(f0, (1,1,-1))
        ec = np.reshape(ec, (1,1,-1))

        f0_converted, ec_converted, coded_sp_converted = model.test(input_pitch=f0, 
                                                      input_mfc=coded_sp,
                                                      input_energy=np.log(ec),
                                                      direction=conversion_direction)
        
        pylab.plot(ec.reshape(-1,), label='processed energy')
        pylab.plot(np.exp(ec_converted.reshape(-1,)), label='converted processed energy')
        pylab.legend()
        
        pylab.figure(), pylab.plot(np.log(ec.reshape(-1,)), label='processed log energy')
        pylab.plot(ec_converted.reshape(-1,), label='converted log energy')
        pylab.legend()

        coded_sp_converted = np.asarray(np.transpose(np.squeeze(coded_sp_converted)), 
                                        np.float64)
        coded_sp_converted = np.ascontiguousarray(coded_sp_converted)
        f0_converted = np.asarray(np.reshape(f0_converted[0], (-1,)), np.float64)
        f0_converted = np.ascontiguousarray(f0_converted)
        f0_converted[f0_z_idx] = 0

        # Pyworld decoding
#        decoded_sp_converted = preproc.world_decode_spectral_envelope(coded_sp=coded_sp_converted, 
#                                                                    fs=sampling_rate)
        
        ec_converted = preprocess_contour(np.exp(ec_converted.reshape(-1,)))
        ec_converted[ec_z_idx] = 1e-04
        
        decoded_sp_converted = np.multiply(sp.T, np.divide(ec_converted.reshape(1,-1), 
                                    ec.reshape(1,-1))**0.5)

        # Normalization of converted features
#        decoded_sp_converted = decoded_sp_converted.T / np.max(decoded_sp_converted)
#        decoded_sp_converted = np.ascontiguousarray(decoded_sp_converted)
        
        wav_transformed = preproc.world_speech_synthesis(f0=f0_converted, 
                                                         decoded_sp=decoded_sp_converted.T, 
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
    
        for file in os.listdir(data_dir):
    
            filepath = os.path.join(data_dir, file)
            
            sr, wav = scwav.read(filepath)
            wav = np.asarray(wav, np.float64)
            wav = normalize_wav(wav)
            assert (sr==sampling_rate)
            
            wav = preproc.wav_padding(wav=wav, sr=sampling_rate, \
                              frame_period=frame_period, multiple=4)
            
            f0, sp, ap = preproc.world_decompose(wav=wav, \
                            fs=sampling_rate, frame_period=frame_period)
            ec = np.reshape(np.sum(sp**2, axis=1), (-1,))
            
            coded_sp = preproc.world_encode_spectral_envelope(sp=sp, \
                                fs=sampling_rate, dim=num_mfcc)
            
            coded_sp = np.expand_dims(coded_sp, axis=0)
            coded_sp = np.transpose(coded_sp, (0,2,1))
            
            f0_z_idx = np.where(f0<10.0)[0]
            ec_z_idx = np.where(ec<1e-04)[0]

            f0 = preprocess_contour(f0)
            ec = preprocess_contour(ec)

            f0 = np.reshape(f0, (1,1,-1))
            ec = np.reshape(ec, (1,1,-1))

            f0_converted, ec_converted, coded_sp_converted = model.test(input_pitch=f0, 
                                                          input_mfc=coded_sp,
                                                          input_energy=np.log(ec),
                                                          direction=conversion_direction)

            coded_sp_converted = np.asarray(np.transpose(np.squeeze(coded_sp_converted)), 
                                            np.float64)
            coded_sp_converted = np.ascontiguousarray(coded_sp_converted)
            f0_converted = np.asarray(np.reshape(f0_converted[0], (-1,)), np.float64)
            f0_converted = np.ascontiguousarray(f0_converted)
            f0_converted[f0_z_idx] = 0

            # Pyworld decoding
#            decoded_sp_converted = preproc.world_decode_spectral_envelope(coded_sp=coded_sp_converted, 
#                                                                         fs=sampling_rate)
            
            ec_converted = preprocess_contour(np.exp(ec_converted.reshape(-1,)))
            ec_converted[ec_z_idx] = 1e-04
            
            decoded_sp_converted = np.multiply(sp.T, np.divide(ec_converted.reshape(1,-1), 
                                    ec.reshape(1,-1))**0.5)

            # Normalization of converted features
#            decoded_sp_converted = decoded_sp_converted.T / np.max(decoded_sp_converted)
#            decoded_sp_converted = np.ascontiguousarray(decoded_sp_converted)
            
            wav_transformed = preproc.world_speech_synthesis(f0=f0_converted, 
                                                             decoded_sp=decoded_sp_converted.T, 
                                                             ap=ap, fs=sampling_rate, 
                                                             frame_period=frame_period)

            wav_transformed = -1 + 2*(wav_transformed - np.min(wav_transformed)) \
                / (np.max(wav_transformed) - np.min(wav_transformed))
            wav_transformed = wav_transformed - np.mean(wav_transformed)
            
            scwav.write(os.path.join(output_dir, os.path.basename(file)), 
                        16000, wav_transformed)
            print('Processed: ' + file)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description = 'Convert Emotion using pre-trained VariationalCycleGAN model.')

    model_dir_default = './model/neu-ang/lp_1e-06_lm_0.1_lmo_1e-06_lrg_0.0001_lrd_1e-06_li_0.05_energy_f0_mwd'
    model_name_default = 'neu-ang_800.ckpt'
    data_dir_default = 'data/evaluation/neu-ang/neutral_5'
    conversion_direction_default = 'A2B'
    output_dir_default = '/home/ravi/Desktop/pitch_energy_wasserstein'
    audio_file_default = '/home/ravi/Desktop/spect-pitch-gan/data/evaluation/neu-ang/neutral_5/1132.wav'

    parser.add_argument('--model_dir', type = str, help='Directory for the pre-trained model.', default=model_dir_default)
    parser.add_argument('--model_name', type = str, help='Filename for the pre-trained model.', default=model_name_default)
    parser.add_argument('--data_dir', type=str, help='Directory for the voices for conversion.', default=data_dir_default)
    parser.add_argument('--conversion_direction', type=str, help='Conversion direction for VCGAN, A2B or B2A', default=conversion_direction_default)
    parser.add_argument('--output_dir', type=str, help='Directory for the converted voices.', default=output_dir_default)
    parser.add_argument('--audio_file', type=str, help='convert a single audio file', default=audio_file_default)

    argv = parser.parse_args()

    model_dir = argv.model_dir
    model_name = argv.model_name
    data_dir = argv.data_dir
    conversion_direction = argv.conversion_direction
    output_dir = argv.output_dir
    audio_file = argv.audio_file
    
    conversion(model_dir=model_dir, model_name=model_name, audio_file=audio_file, 
               data_dir=data_dir, conversion_direction=conversion_direction, 
               output_dir=output_dir, embedding=True, only_energy=True)


