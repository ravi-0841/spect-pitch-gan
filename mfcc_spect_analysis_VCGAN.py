import argparse
import os
import numpy as np
import librosa
import scipy.io.wavfile as scwav
import scipy.signal as scisig
import tensorflow as tf
import utils.preprocess as preproc
import pylab
import librosa
import scipy

from numpy.fft import rfft, irfft
from utils.helper import smooth, generate_interpolation, mfcc_to_spectrum
from nn_models.model_separate_discriminate_id import VariationalCycleGAN
from scipy.optimize import nnls, fmin_l_bfgs_b
from concurrent.futures import ProcessPoolExecutor
from functools import partial


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


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


def _hz2mel(hz):
    """Convert a value in Hertz to Mels
    :param hz: a value in Hz. This can also be a numpy array, conversion proceeds element-wise.
    :returns: a value in Mels. If an array was passed in, an identical sized array is returned.
    """
    return 2595 * np.log10(1 + hz / 700.)


def _mel2hz(mel):
    """Convert a value in Mels to Hertz
    :param mel: a value in Mels. This can also be a numpy array, conversion proceeds element-wise.
    :returns: a value in Hertz. If an array was passed in, an identical sized array is returned.
    """
    return 700 * (10 ** (mel / 2595.0) - 1)


def tuanad_encode_mcep(spec: np.ndarray, n0: int = 20, fs: int = 16000, 
                       lowhz=0, highhz=8000):
    """
    Warp magnitude spectrum with Mel-scale
    Then, cepstrum analysis with order of n0
    Spec is magnitude spectrogram (N x D) array
    """

    lowmel = _hz2mel(lowhz)
    highmel = _hz2mel(highhz)
    """return the real cepstrum X is N x D array; N frames and D dimensions"""
    Xl = np.log(spec)
    D = spec.shape[1]
    melpoints = np.linspace(lowmel, highmel, D)
    bin = np.floor(((D - 1) * 2 + 1) * _mel2hz(melpoints) / fs)
    Xml = np.array([np.interp(bin, np.arange(D), s)
                    for s in Xl])  #
    Xc = irfft(Xml)  # Xl is real, not complex
    return Xc[:, :n0]


def tuanad_decode_mcep(cepstrum: np.ndarray, fft_size:int):
    """
    Compute magnitude spectrum from mcep Tuanad implementation
    """
    lowmel = _hz2mel(0)
    highmel = _hz2mel(8000)
    n0 = cepstrum.shape[1]
    Yc = np.zeros((cepstrum.shape[0], fft_size))
    Yc[:, :n0] = cepstrum
    Yc[:, :-n0:-1] = Yc[:, 1:n0]
    Yl = rfft(Yc).real
    melpoints = np.linspace(lowmel, highmel, int(fft_size // 2 + 1))
    bin = np.floor(fft_size * _mel2hz(melpoints) / 16000)
    Yl = np.array([np.interp(np.arange(int(fft_size // 2 + 1)), bin, s)
                   for s in Yl])
    return np.exp(Yl)


if __name__ == '__main__':
    
    tf.reset_default_graph()
    
    num_mfcc = 23
    num_pitch = 1
    sampling_rate = 16000
    frame_period = 5.0
    num_mels = 128
    n_fft = 1024

    parser = argparse.ArgumentParser(description = 'Convert Emotion using pre-trained VariationalCycleGAN model.')

    model_dir = '/home/ravi/Desktop/spect-pitch-gan/model/neu-ang/lp_1e-05_lm_1.0_lmo_1e-06_li_0.5_pre_trained_id'
    model_name = 'neu-ang_1000.ckpt'
    conversion_direction = 'A2B'
    audio_file = '/home/ravi/Desktop/spect-pitch-gan/data/evaluation/neu-ang/neutral_5/1081.wav'
    
    model = VariationalCycleGAN(dim_mfc=num_mfcc, dim_pitch=num_pitch, mode='test')
    model.load(filepath=os.path.join(model_dir, model_name))

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
    
    f0 = scisig.medfilt(f0, kernel_size=3)
    z_idx = np.where(f0<10.0)[0]
    f0 = generate_interpolation(f0)
    f0 = smooth(f0, window_len=13)
    f0 = np.reshape(f0, (1,1,-1))

    f0_converted, coded_sp_converted = model.test(input_pitch=f0, 
                                                  input_mfc=coded_sp, 
                                                  direction=conversion_direction)
    
    
    """
    Pyworld conversion of mfcc to spectrum
    """
    coded_sp_converted = np.asarray(np.transpose(coded_sp_converted[0]), np.float64)
    coded_sp_converted = np.ascontiguousarray(coded_sp_converted)
    f0_converted = np.asarray(np.reshape(f0_converted[0], (-1,)), np.float64)
    f0_converted = np.ascontiguousarray(f0_converted)
    f0_converted[z_idx] = 0
    
    decoded_sp_pyworld = preproc.world_decode_spectral_envelope(coded_sp=coded_sp_converted, 
                                                                 fs=sampling_rate)
    decoded_sp_pyworld = smooth_spectrum(decoded_sp_pyworld)
    print('Pyworld decoded')
    
    """
    Librosa conversion of mfcc to spectrum
    """
    decoded_sp_librosa = mfcc_to_spectrum(coded_sp_converted, axis=1, 
                                         sr=sampling_rate)
    print('Librosa decoded')
    
    """
    Tuanad conversion of mfcc to spectrum
    """
    encoded_sp_tuanad = tuanad_encode_mcep(sp, n0=num_mfcc)
    decoded_sp_tuanad = tuanad_decode_mcep(coded_sp_converted, fft_size=n_fft)
    print('Tuanad decoded')

    """
    Manual conversion of mfcc to spectrum
    """
    filters = librosa.filters.mel(sr=sampling_rate, n_fft=n_fft, 
                                  n_mels=num_mels, htk=True)
    log_filter_energy = scipy.fftpack.idct(coded_sp_converted, axis=1, 
                                     type=2, norm='ortho', n=num_mels)
    filter_energy = _db_to_power(log_filter_energy)
    
    decoded_sp_manual = np.transpose(np.power(nnls_lbfgs_block(np.asarray(filters, np.float64), 
                                                   filter_energy.T), 1/2.0))
    print('Manual decoded')

    # Plot the spectrum
    pylab.figure(), pylab.subplot(221)
    pylab.imshow(decoded_sp_librosa.T / np.max(decoded_sp_librosa)), pylab.colorbar(), pylab.title('Librosa')
    pylab.subplot(222)
    pylab.imshow(decoded_sp_manual.T/np.max(decoded_sp_manual)), pylab.colorbar(), pylab.title('Manual')
    pylab.subplot(223)
    pylab.imshow(decoded_sp_pyworld.T/np.max(decoded_sp_pyworld)), pylab.colorbar(), pylab.title('Pyworld')
    pylab.subplot(224)
    pylab.imshow(decoded_sp_tuanad.T/np.max(decoded_sp_tuanad)), pylab.colorbar(), pylab.title('Tuanad')

    wav_transformed = preproc.world_speech_synthesis(f0=f0_converted, 
                                                     decoded_sp=decoded_sp_tuanad/np.max(decoded_sp_tuanad), 
                                                     ap=ap, fs=sampling_rate, 
                                                     frame_period=frame_period)
    scwav.write(os.path.join('/home/ravi/Desktop', 
                             os.path.basename(audio_file)), 
                            sampling_rate, wav_transformed)
    print('Processed: ' + audio_file)
    