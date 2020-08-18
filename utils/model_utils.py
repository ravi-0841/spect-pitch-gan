import tensorflow as tf
import os
import random
import numpy as np

def _hz2mel(hz):
    return 2595 * np.log10(1 + hz/700.)

def _mel2hz(mel):
    return 700 * (10 ** (mel/2595.0) - 1)

def _log10(x):
    return tf.divide(tf.log(x), tf.log(10.0))

def _power_to_db(s):
    return 20*_log10(s)

def _sliding_windows(template, size):
    template = np.asarray(template)
    p = np.zeros(size-1, dtype=template.dtype)
    b = np.concatenate((p, template, p))
    s = b.strides[0]
    strided = np.lib.stride_tricks.as_strided
    return strided(b[size-1:], shape=(size,len(template)+size-1), strides=(-s,s))

def _interp_mat_mel2hz(sr=16000, n_fft=1024):
    lowmel = _hz2mel(0)
    highmel = _hz2mel(sr/2)
    mel_points = np.linspace(lowmel, highmel, n_fft//2 + 1)
    values_available_at = _mel2hz(mel_points)
    values_desired_at = np.linspace(0, sr/2, n_fft//2 + 1)
    F_inv = np.zeros((n_fft//2 + 1, n_fft//2 + 1))
    for f in range(n_fft//2 + 1):
        cf = values_desired_at[f]
        for i in range(n_fft//2):
            if (cf >= values_available_at[i]) and (cf <= values_available_at[i+1]):
                bin_width = values_available_at[i+1] - values_available_at[i]
                ldist = 1 - ((cf - values_available_at[i]) / bin_width)
                rdist = 1 - ((values_available_at[i+1] - cf) / bin_width)
                F_inv[f,i] = ldist
                F_inv[f,i+1] = rdist
                break
    F_inv[-1,-1] = 1
    return np.asarray(F_inv, np.float32)

def delta_matrix(time_steps=128):
    delta_matrix = _sliding_windows([1,0,-1], 130)
    delta_matrix = delta_matrix[1:-1, 2:-2]
    return np.asarray(delta_matrix, np.float32)

def l1_loss(y, y_hat):

    return tf.reduce_mean(tf.abs(y - y_hat))

def l2_loss(y, y_hat):

    return tf.reduce_mean(tf.square(y - y_hat))

def wg_loss(y, y_hat):
    
    return tf.reduce_mean(y_hat) - tf.reduce_mean(y)

def cross_entropy_loss(logits, labels):

    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = logits, labels = labels))

def mcd_loss(y, y_hat):
    y = tf.squeeze(y)
    y_hat = tf.squeeze(y_hat)
    mcd_loss = tf.reduce_mean(tf.pow(tf.reduce_sum(tf.pow(y - y_hat, 2), axis=0), 0.5))
    return mcd_loss

def spectral_loss(y, y_hat, pad_right=490, fft_size=1024.0, interp_mat=None):
    if interp_mat is None:
        interp_mat = _interp_mat_mel2hz()
    
    y = tf.squeeze(y)
    y_hat = tf.squeeze(y_hat)
    extend_y = tf.pad(tf.transpose(y), [[0,0], [0,pad_right]], 'constant')
    extend_y_hat = tf.pad(tf.transpose(y_hat), [[0,0], [0,pad_right]], 'constant')
    
    idct_y = tf.signal.idct(extend_y*tf.math.sqrt(fft_size), type=2, norm='ortho')
    idct_y_hat = tf.signal.idct(extend_y_hat*tf.math.sqrt(fft_size), type=2, norm='ortho')

    mel2hz_y = tf.transpose(tf.matmul(interp_mat, idct_y, transpose_b=True))
    mel2hz_y_hat = tf.transpose(tf.matmul(interp_mat, idct_y_hat, transpose_b=True))

    spect_y = tf.math.exp(mel2hz_y)
    spect_y_hat = tf.math.exp(mel2hz_y_hat)
    
    spect_y = spect_y / tf.reduce_max(spect_y)
    spect_y_hat = spect_y_hat / tf.reduce_max(spect_y_hat)

    spect_y = tf.add(spect_y, 1e-10)
    spect_y_hat = tf.add(spect_y_hat, 1e-10)

    spect_y = _power_to_db(tf.pow(spect_y, 2))
    spect_y_hat = _power_to_db(tf.pow(spect_y_hat, 2))

    spect_y = tf.divide((spect_y - tf.reduce_min(spect_y)), 
            (tf.reduce_max(spect_y) - tf.reduce_min(spect_y)))
    spect_y_hat = tf.divide((spect_y_hat - tf.reduce_min(spect_y_hat)), 
            (tf.reduce_max(spect_y_hat) - tf.reduce_min(spect_y_hat)))

    return tf.reduce_mean(tf.abs(spect_y - spect_y_hat))

def mfcc_derivative_loss(y, y_hat, derivative_op=None):
    """
    Expects y/y_hat to be of shape batch_size x features_dim x time_steps (default=128)
    """
    if derivative_op is None:
        derivative_op = delta_matrix()
    y_derivative = tf.matmul(y, derivative_op)
    y_hat_derivative = tf.matmul(y_hat, derivative_op)
    return tf.reduce_mean(tf.abs(y_derivative - y_hat_derivative))

def eval_kernel(kernel1, kernel2):
    exp_kernel = tf.exp(-1*tf.reduce_sum(tf.pow(kernel1 - kernel2, 2)))
    return exp_kernel

















































