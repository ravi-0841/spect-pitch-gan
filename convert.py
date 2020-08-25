import argparse
import os
import numpy as np
import librosa
import scipy.io.wavfile as scwav
import scipy.signal as scisig
import pylab

import utils.preprocess as preproc
from utils.helper import smooth, generate_interpolation
from nn_models.model_embedding_wasserstein import VariationalCycleGAN
from encoder_decoder import AE


num_mfcc = 23
num_pitch = 1
sampling_rate = 16000
frame_period = 5.0


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


def conversion(model_dir=None, model_name=None, audio_file=None, 
               data_dir=None, conversion_direction=None, output_dir=None):
    
    ae_model = AE(dim_mfc=num_mfcc)
    ae_model.load(filename='./model/AE_cmu_pre_trained_noise_std_1.ckpt')
    model = VariationalCycleGAN(dim_mfc=1, dim_pitch=1, mode='test')
    model.load(filepath=os.path.join(model_dir, model_name))
    
    if audio_file is not None:
        wav, sr = librosa.load(audio_file, sr=sampling_rate, mono=True)
        assert (sr==sampling_rate)
        wav = preproc.wav_padding(wav=wav, sr=sampling_rate, \
                          frame_period=frame_period, multiple=4)
        f0, sp, ap = preproc.world_decompose(wav=wav, \
                        fs=sampling_rate, frame_period=frame_period)
        coded_sp = preproc.world_encode_spectral_envelope(sp=sp, \
                            fs=sampling_rate, dim=num_mfcc)
        
        coded_sp = np.expand_dims(coded_sp, axis=0)
        coded_sp = np.transpose(coded_sp, (0,2,1))
        sp_embedding = ae_model.get_embedding(mfc_features=coded_sp)
        
        f0 = scisig.medfilt(f0, kernel_size=3)
        z_idx = np.where(f0<10.0)[0]
        f0 = generate_interpolation(f0)
        f0 = smooth(f0, window_len=13)
        f0 = np.reshape(f0, (1,1,-1))

        f0_converted, coded_sp_converted = model.test(input_pitch=f0, 
                                                      input_mfc=sp_embedding, 
                                                      direction=conversion_direction)
        

        coded_sp_converted = ae_model.get_mfcc(embeddings=coded_sp_converted)
        coded_sp_converted = np.asarray(np.transpose(coded_sp_converted[0]), np.float64)
        coded_sp_converted = np.ascontiguousarray(coded_sp_converted)
        f0_converted = np.asarray(np.reshape(f0_converted[0], (-1,)), np.float64)
        f0_converted = np.ascontiguousarray(f0_converted)
        f0_converted[z_idx] = 0
        
        decoded_sp_converted = preproc.world_decode_spectral_envelope(coded_sp=coded_sp_converted, 
                                                                     fs=sampling_rate)
        # Normalization of converted features
        decoded_sp_converted = decoded_sp_converted / np.max(decoded_sp_converted)
        wav_transformed = preproc.world_speech_synthesis(f0=f0_converted, 
                                                         decoded_sp=decoded_sp_converted, 
                                                         ap=ap, fs=sampling_rate, 
                                                         frame_period=frame_period)
        scwav.write(os.path.join('/home/ravi/Desktop', 
                                 os.path.basename(audio_file)), 
                                sampling_rate, wav_transformed)
        print('Processed: ' + audio_file)
        
    else:
        os.makedirs(output_dir, exist_ok=True)
    
        for file in os.listdir(data_dir):
    
            filepath = os.path.join(data_dir, file)
            wav, sr = librosa.load(filepath, sr=sampling_rate, mono=True)
            assert (sr==sampling_rate)
            wav = preproc.wav_padding(wav=wav, sr=sampling_rate, \
                              frame_period=frame_period, multiple=4)
            f0, sp, ap = preproc.world_decompose(wav=wav, \
                            fs=sampling_rate, frame_period=frame_period)
            coded_sp = preproc.world_encode_spectral_envelope(sp=sp, \
                                fs=sampling_rate, dim=num_mfcc)
            
            coded_sp = np.expand_dims(coded_sp, axis=0)
            coded_sp = np.transpose(coded_sp, (0,2,1))
            sp_embedding = ae_model.get_embedding(mfc_features=coded_sp)
            
            f0 = scisig.medfilt(f0, kernel_size=3)
            z_idx = np.where(f0<10.0)[0]
            f0 = generate_interpolation(f0)
            f0 = smooth(f0, window_len=13)
            f0 = np.reshape(f0, (1,1,-1))
    
            f0_converted, coded_sp_converted = model.test(input_pitch=f0, 
                                                          input_mfc=sp_embedding, 
                                                          direction=conversion_direction)
            
            coded_sp_converted = ae_model.get_mfcc(embeddings=coded_sp_converted)
            coded_sp_converted = np.asarray(np.transpose(coded_sp_converted[0]), np.float64)
            coded_sp_converted = np.ascontiguousarray(coded_sp_converted)
            f0_converted = np.asarray(np.reshape(f0_converted[0], (-1,)), np.float64)
            f0_converted = np.ascontiguousarray(f0_converted)
            f0_converted[z_idx] = 0
            
            # Mixing the mfcc features
#            coded_sp_converted = np.max(coded_sp) \
#                * (coded_sp_converted - np.min(coded_sp_converted)) \
#                / (np.max(coded_sp_converted) - np.min(coded_sp_converted)) + np.min(coded_sp)
            coded_sp_converted = 0.6*coded_sp_converted + 0.4*np.transpose(np.squeeze(coded_sp))
            
            # Pyworld decoding
            decoded_sp_converted = preproc.world_decode_spectral_envelope(coded_sp=coded_sp_converted, 
                                                                         fs=sampling_rate)
            
            # Mixing the decoded sp and input sp
#            decoded_sp_converted = decoded_sp_converted / np.max(decoded_sp_converted)
#            sp = sp / np.max(sp)
#            decoded_sp_converted = 0.5*decoded_sp_converted + 0.5*sp
            
            # Normalization of converted features
#            decoded_sp_converted = decoded_sp_converted / np.max(decoded_sp_converted)
            wav_transformed = preproc.world_speech_synthesis(f0=f0_converted, 
                                                             decoded_sp=decoded_sp_converted, 
                                                             ap=ap, fs=sampling_rate, 
                                                             frame_period=frame_period)
            
            wav_transformed = (wav_transformed - np.min(wav_transformed)) \
                / (np.max(wav_transformed) - np.min(wav_transformed))
            wav_transformed = wav_transformed - np.mean(wav_transformed)
            
            scwav.write(os.path.join(output_dir, 'spect_mixing_denoised_'+os.path.basename(file)), 
                        sampling_rate, wav_transformed)
            print('Processed: ' + file)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description = 'Convert Emotion using pre-trained VariationalCycleGAN model.')

    model_dir_default = './model/neu-ang/lp_1e-05_lm_1.0_lmo_1e-06_li_0.5_pre_trained_embedding_wasserstein'
    model_name_default = 'neu-ang_1000.ckpt'
    data_dir_default = 'data/evaluation/neu-ang/neutral_5'
    conversion_direction_default = 'A2B'
    output_dir_default = '/home/ravi/Desktop/converted_emotion_AE_wasserstein'
    audio_file_default = None

    parser.add_argument('--model_dir', type = str, help = 'Directory for the pre-trained model.', default=model_dir_default)
    parser.add_argument('--model_name', type = str, help = 'Filename for the pre-trained model.', default=model_name_default)
    parser.add_argument('--data_dir', type = str, help = 'Directory for the voices for conversion.', default=data_dir_default)
    parser.add_argument('--conversion_direction', type = str, help = 'Conversion direction for VCGAN, A2B or B2A', default=conversion_direction_default)
    parser.add_argument('--output_dir', type = str, help = 'Directory for the converted voices.', default=output_dir_default)
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
               output_dir=output_dir)


