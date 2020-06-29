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
import argparse

from numpy.fft import rfft, irfft
from utils.helper import smooth, generate_interpolation, mfcc_to_spectrum
from nn_models.model_separate_discriminate_id import VariationalCycleGAN
from scipy.optimize import nnls, fmin_l_bfgs_b
from scipy.signal import butter, lfilter, freqz, filtfilt
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


def _butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a


def _butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = _butter_lowpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data)
    return y


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


def _interp_matrix_hz2mel(sr=16000, n_fft=1024):
    lowmel = _hz2mel(0)
    highmel = _hz2mel(sr/2)
    mel_points = np.linspace(lowmel, highmel, n_fft//2 + 1)
    center_freq_points = _mel2hz(mel_points)
    bin_width = sr / n_fft
    freq_ends = np.arange(0, sr/2 + bin_width, bin_width)
    F = np.zeros((n_fft//2+1, n_fft//2+1))
    for m in range(n_fft//2+1):
        mf = center_freq_points[m]
        for i in range(n_fft//2):
            if (mf >= freq_ends[i]) and (mf < freq_ends[i+1]):
                ldist = 1 - ((mf - freq_ends[i]) / bin_width)
                rdist = 1 - ((freq_ends[i+1] - mf) / bin_width)
                F[m, i] = ldist
                F[m, i+1] = rdist
                break
    F[-1,-1] = 1
    return F


def _interp_matrix_mel2hz(sr=16000, n_fft=1024):
    lowmel = _hz2mel(0)
    highmel = _hz2mel(sr/2)
    mel_points = np.linspace(lowmel, highmel, n_fft//2 + 1)
    values_available_at = _mel2hz(mel_points)
    values_desired_at = np.linspace(0, sr/2, n_fft//2 + 1)
    F_inv = np.zeros((n_fft//2+1, n_fft//2+1))
    for f in range(n_fft//2+1):
        cf = values_desired_at[f]
        for i in range(n_fft//2):
            if (cf >= values_available_at[i]) and (cf <= values_available_at[i+1]):
                bin_width = values_available_at[i+1] - values_available_at[i]
                ldist = 1 - ((cf - values_available_at[i]) / bin_width)
                rdist = 1 - ((values_available_at[i+1] - cf) / bin_width)
                F_inv[f, i] = ldist
                F_inv[f, i+1] = rdist
                break
    F_inv[-1,-1] = 1
    return F_inv


def _f0_interp(f0, s):
#    bin_sep = int(np.ceil(f0 / ((sampling_rate/2)/(n_fft//2 + 1))))
#    sampling_x = np.asarray(np.floor(np.arange(0, (n_fft//2 + 1), bin_sep)), np.int)
#    sampling_y = s[sampling_x]
#    interp_x = np.arange(0, (n_fft//2 + 1), 1)
#    interp_y = np.interp(interp_x, sampling_x, sampling_y)

    x_org = np.arange(0, 8000, f0)
    x_org = x_org*n_fft / sampling_rate
    x_org = np.asarray(np.ceil(x_org), np.int32)
    y_org = s[x_org]
    interp_x = np.arange(0, (n_fft//2 + 1), 1)
    interp_y = np.interp(interp_x, x_org, y_org)

    return interp_y


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


def tuanad_encode_mcep(spec: np.ndarray, n0: int = 20, fs: int = 16000, 
                       lowhz=0, highhz=8000):
    """
    Warp magnitude spectrum with Mel-scale
    Then, cepstrum analysis with order of n0
    Spec is magnitude spectrogram (N x D) array
    """
    interp_mat = _interp_matrix_hz2mel()
    Xml = np.dot(interp_mat, np.log(spec.T)).T
    Xc = scipy.fftpack.dct(Xml, axis=1, norm='ortho') / np.sqrt(n_fft)
    return Xc[:, :n0]


def tuanad_decode_mcep(cepstrum: np.ndarray, n_fft:int):
    """
    cepstrum: array TxD, T - timeframes and D - fft_size//2 + 1
    """
    interp_mat = _interp_matrix_mel2hz()
    Yl = scipy.fftpack.idct(cepstrum*np.sqrt(n_fft), axis=1, 
                            n=(n_fft//2 + 1), norm='ortho')
    Yl = np.dot(interp_mat, Yl.T).T

    return np.exp(Yl)


def low_pass_filt(S, cutoff_freq=10, fs=32):
    S_filtered = np.empty((0,S.shape[-1]))
    for i in S:
        i_filtered = _butter_lowpass_filter(i, cutoff_freq, fs)
        S_filtered = np.concatenate((S_filtered, i_filtered.reshape(1,-1)))
    return S_filtered


def f0_spect_consistency(f0, spect):
    nz_idx = np.where(f0>10.0)
    f0 = f0[nz_idx]
    spect = spect[nz_idx]
    spect = _power_to_db(spect)
    interp_spect = np.array([_f0_interp(f0[i], spect[i]) for i in range(spect.shape[0])])
    return (spect, interp_spect, np.mean(np.sum(np.abs(spect - interp_spect), axis=1)))


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
    audio_file = '/home/ravi/Desktop/spect-pitch-gan/data/evaluation/neu-ang/neutral_5/1175.wav'
    
    parser.add_argument('--audio_file', type=str, help='audio file to convert', default=audio_file)
    argv = parser.parse_args()

    model = VariationalCycleGAN(dim_mfc=num_mfcc, dim_pitch=num_pitch, mode='test')
    model.load(filepath=os.path.join(model_dir, model_name))

    wav, sr = librosa.load(argv.audio_file, sr=sampling_rate, mono=True)
    assert (sr==sampling_rate)
    wav = preproc.wav_padding(wav=wav, sr=sampling_rate, \
                      frame_period=frame_period, multiple=4)
    f0, sp, ap = preproc.world_decompose(wav=wav, \
                    fs=sampling_rate, frame_period=frame_period)
    coded_sp = preproc.world_encode_spectral_envelope(sp=sp, \
                        fs=sampling_rate, dim=num_mfcc)
#    coded_sp = tuanad_encode_mcep(spec=sp, n0=num_mfcc, fs=sampling_rate)
#    coded_sp = preproc.encode_raw_spectrum(spectrum=sp, axis=1, 
#                                           dim_mfc=num_mfcc)
    
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
    coded_sp = np.asarray(np.transpose(coded_sp[0]), np.float64)
    coded_sp_converted = np.asarray(np.transpose(coded_sp_converted[0]), np.float64)
    coded_sp_converted = np.copy(coded_sp_converted, order='C')
    f0_converted = np.asarray(np.reshape(f0_converted[0], (-1,)), np.float64)
    f0_converted = np.copy(f0_converted, order='C')
    f0_converted[z_idx] = 0
    
    decoded_sp_pyworld = preproc.world_decode_spectral_envelope(coded_sp=coded_sp_converted, 
                                                                 fs=sampling_rate)
#    decoded_sp_pyworld = preproc.decode_raw_spectrum(linear_mfcc=coded_sp_converted, 
#                                                     axis=1, n_fft=n_fft)
#    decoded_sp_pyworld = smooth_spectrum(decoded_sp_pyworld)
    print('Pyworld decoded')
    

    spect_conv, interp_spect_conv, error_conv = f0_spect_consistency(f0_converted, decoded_sp_pyworld)
    spect, interp_spect, error_orig = f0_spect_consistency(f0.reshape(-1,), sp)
    print('Original mismatch- {} and reconstructed mismatch- {}'.format(error_orig, error_conv))


#    interp_mat_mel2hz = np.asarray(_interp_matrix_mel2hz(sr=16000, n_fft=1024), 
#                                   np.float32)
#    mfcc_tensor = tf.placeholder(dtype=tf.float32, shape=(None, 23))
#    mfcc_extend_tensor = tf.pad(mfcc_tensor, [[0,0],[0,513-23]], 'constant')
#    idct_tensor = tf.signal.idct(mfcc_extend_tensor*np.sqrt(1024), 
#                                 type=2, norm='ortho')
#    mel2hz_tensor = tf.transpose(tf.matmul(interp_mat_mel2hz, idct_tensor, 
#                                       transpose_b=True))
#    spect_tensor = tf.math.exp(mel2hz_tensor)
#    
#    
#    with tf.Session() as sess:
#        decoded_sp_tensorflow = sess.run(spect_tensor, 
#                                         feed_dict={mfcc_tensor:np.asarray(coded_sp_converted,
#                                                                           np.float32)})
#    
#    lowmel = _hz2mel(0)
#    highmel = _hz2mel(sampling_rate/2)
#    melpoints = np.linspace(lowmel, highmel, int(n_fft // 2 + 1))
#    bin = np.floor(n_fft * _mel2hz(melpoints) / sampling_rate)
#    z = np.array([np.interp(np.arange(int(n_fft // 2 + 1)), bin, s) for s in z])
#    z = np.exp(z)
#    z_nz, interp_znz, error_znz = f0_spect_consistency(f0_converted, z)
#    print('Tensorflow mismatch- {}'.format(error_znz))

    """
    Plotting random spectrum slices
    """
#    for i in range(10):
#        q = np.random.randint(0, min([sp.shape[0], decoded_sp_pyworld.shape[0]]))
#        pylab.figure(), pylab.subplot(121), pylab.plot(coded_sp[q,:], 'g', label='natural')
#        pylab.title('Non Converted'), pylab.legend()
#        pylab.subplot(122), pylab.plot(coded_sp_converted[q,:], 'r', label='converted')
#        pylab.title('Converted'), pylab.legend()
#        pylab.suptitle('Frame %d' % q)

    """
    Tuanad conversion of mfcc to spectrum
    """
#    encoded_sp_tuanad = tuanad_encode_mcep(sp, n0=num_mfcc)
#    decoded_sp_tuanad = tuanad_decode_mcep(coded_sp_converted, n_fft=n_fft)
#    print('Tuanad decoded')

    """
    Manual conversion of mfcc to spectrum
    """
#    filters = librosa.filters.mel(sr=sampling_rate, n_fft=n_fft, 
#                                  n_mels=num_mels, htk=True)
#    log_filter_energy = scipy.fftpack.idct(coded_sp_converted, axis=1, 
#                                     type=2, norm='ortho', n=num_mels)
#    filter_energy = _db_to_power(log_filter_energy)
#    
#    decoded_sp_manual = np.transpose(np.power(nnls_lbfgs_block(np.asarray(filters, np.float64), 
#                                                   filter_energy.T), 1/2.0))
#    print('Manual decoded')
#
#    # Plot the spectrum
#    pylab.figure(), pylab.subplot(221)
#    pylab.imshow(decoded_sp_librosa.T / np.max(decoded_sp_librosa)), pylab.colorbar(), pylab.title('Librosa')
#    pylab.subplot(222)
#    pylab.imshow(decoded_sp_manual.T/np.max(decoded_sp_manual)), pylab.colorbar(), pylab.title('Manual')
#    pylab.subplot(223)
#    pylab.imshow(decoded_sp_pyworld.T/np.max(decoded_sp_pyworld)), pylab.colorbar(), pylab.title('Pyworld')
#    pylab.subplot(224)
#    pylab.imshow(decoded_sp_tuanad.T/np.max(decoded_sp_tuanad)), pylab.colorbar(), pylab.title('Tuanad')


    """
    Synthesizing Speech Pyworld
    """
#    decoded_sp_pyworld = np.copy(decoded_sp_pyworld, order='C')
#    wav_transformed = preproc.world_speech_synthesis(f0=f0_converted, 
#                                                     decoded_sp=decoded_sp_pyworld/np.max(decoded_sp_pyworld), 
#                                                     ap=ap, fs=sampling_rate, 
#                                                     frame_period=frame_period)
#    scwav.write(os.path.join('/home/ravi/Desktop/', 
#                             'pyworld_'+os.path.basename(audio_file)), 
#                            sampling_rate, wav_transformed)
#    print('Processed: ' + audio_file)

    """
    Synthesizing Speech Tuanad
    """
#    decoded_sp_tuanad = np.copy(decoded_sp_tuanad, order='C')
#    wav_transformed = preproc.world_speech_synthesis(f0=f0_converted, 
#                                                     decoded_sp=decoded_sp_tuanad/np.max(decoded_sp_tuanad), 
#                                                     ap=ap, fs=sampling_rate, 
#                                                     frame_period=frame_period)
#    scwav.write(os.path.join('/home/ravi/Desktop/', 
#                             'tuanad_'+os.path.basename(audio_file)), 
#                            sampling_rate, wav_transformed)
#    print('Processed: ' + audio_file)

    """
    Synthesizing Speech Tensorflow
    """
#    decoded_sp_tensorflow = np.copy(np.asarray(decoded_sp_tensorflow, np.float64), 
#                                order='C')
#    wav_transformed = preproc.world_speech_synthesis(f0=f0_converted, 
#                                                     decoded_sp=decoded_sp_tensorflow/np.max(decoded_sp_tensorflow), 
#                                                     ap=ap, fs=sampling_rate, 
#                                                     frame_period=frame_period)
#    scwav.write(os.path.join('/home/ravi/Desktop/', 
#                             'tensorflow_'+os.path.basename(audio_file)), 
#                            sampling_rate, wav_transformed)
#    print('Processed: ' + audio_file)
    
    
######################################################################################################


