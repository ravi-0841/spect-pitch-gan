#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 30 20:12:05 2021

@author: ravi
"""
import pylab

import scipy.io as scio
import numpy as np
import tensorflow as tf

def dHr_tan(x, p, Css, kernel):
    """
    Computes the small displacement of x and p
    """
#    input_shape = x.get_shape().as_list()
    n = tf.cast(tf.shape(x)[1], dtype=tf.float32)#input_shape[1]
    
    x_repeated_cols = tf.einsum('ijk,kl->ijl', x, tf.ones([1,n], dtype=tf.float32))
    x_repeated_rows = tf.einsum('lj,ijk->ilk',tf.ones([n,1],dtype=tf.float32), \
                                tf.transpose(x, perm=[0,2,1]))
    
    x_repeated_diff = tf.subtract(x_repeated_cols, x_repeated_rows)
    S = tf.add(Css, tf.divide(tf.square(x_repeated_diff), kernel[0,1]**2))
    A = tf.exp(tf.multiply(-1.0,S))
    B = tf.divide(tf.multiply(-1.0,A), kernel[0,1]**2)
    dxHr = tf.einsum('ijk,ikl->ijl', A, p)
    
    C = x_repeated_diff
    prod_BC = tf.multiply(B, C)
    prod_PP = tf.einsum('ijk,ikl->ijl', p, tf.transpose(p, perm=[0,2,1]))
    dpHr = tf.multiply(2.0, tf.reduce_sum(tf.multiply(prod_BC, prod_PP), axis=2, \
                               keepdims=True))
    return dxHr, dpHr

def fdh(x, p, Css, h, kernel):
    """
    compute displacement for step size h
    """
    dx, dp = dHr_tan(x, p, Css, kernel)
    kx = tf.add(x, tf.multiply(h, dx))
    kp = tf.subtract(p, tf.multiply(h, dp))
    return kx, kp

def forward_tan(x, p, kernel):
    """
    Expecting input x to be of shape [#batch, 1, #length]
    Expecting momenta p to be of the shape[#batch, 1, #length]
    """
    x = tf.transpose(x, perm=[0,2,1]) # converts x to shape[#batch, #length, 1]
    p = tf.transpose(p, perm=[0,2,1]) # converts p to shape[#batch, #length, 1]
    input_shape = tf.cast(tf.shape(x), dtype=tf.float32)#x.get_shape().as_list()
    time_axis = tf.expand_dims(tf.range(start=0, limit=input_shape[1], dtype=tf.float32), axis=-1)
    repeated_cols = tf.matmul(time_axis, tf.ones_like(tf.transpose(time_axis,perm=[1,0])))
    repeated_rows = tf.matmul(tf.ones_like(time_axis), tf.transpose(time_axis, perm=[1,0]))
    Css = tf.divide(tf.square(repeated_cols -  repeated_rows), (kernel[0,0]**2))
    
    dt = 1.0 / 3.0
    x_evol = x
    p_evol = p
    
    x2, p2 = fdh(x_evol, p_evol, Css, dt/2, kernel)
    x3, p3 = fdh(x2, p2, Css, dt, kernel)
    
    #-----------------------------------------------------------------------------------------------------
    """
    Manually iterating over 3 time steps
    """
    x_evol = tf.add(tf.subtract(x3,x2),x_evol)
    p_evol = tf.add(tf.subtract(p3,p2),p_evol)
    
    x2, p2 = fdh(x_evol, p_evol, Css, dt/2, kernel)
    x3, p3 = fdh(x2, p2, Css, dt, kernel)
    
    x_evol = tf.add(tf.subtract(x3,x2),x_evol)
    p_evol = tf.add(tf.subtract(p3,p2),p_evol)
    
    x2, p2 = fdh(x_evol, p_evol, Css, dt/2, kernel)
    x3, p3 = fdh(x2, p2, Css, dt, kernel)
    #------------------------------------------------------------------------------------------------------

    x_final = tf.add(tf.subtract(x3,x2), x_evol)
    p_final = tf.add(tf.subtract(p3,p2),p_evol) 
    return tf.transpose(x_final, perm=[0,2,1]), tf.transpose(p_final, perm=[0,2,1]), x_evol, p_evol

if __name__ == "__main__":
    tf.reset_default_graph()
    data = scio.loadmat('/home/ravi/Desktop/Interspeech-2019/momentum-warping/data/neu-ang/mom-valid.mat')
    src_feat = np.asarray(data['src_f0_feat'], np.float64)
    tar_feat = np.asarray(data['tar_f0_feat'], np.float64)
    mom_pitch = np.asarray(data['momentum_pitch'], np.float64)
    src_feat[np.where(src_feat<=0)] = 1e-1
    tar_feat[np.where(tar_feat<=0)] = 1e-1
    q = 13362 #np.random.randint(0, src_feat.shape[0])
    
    X = tf.placeholder(dtype=tf.float32, shape=[None, 1, None], name="input_curve")
    P = tf.placeholder(dtype=tf.float32, shape=[None, 1, None], name="momenta")
    K = tf.placeholder(dtype=tf.float32, shape=[1,2], name="kernel")
    
    warped_curve, final_momenta, intermed_warped, intermed_momenta = forward_tan(X, P, K)
#    grads = tf.gradients(ys=warped_curve, xs=P, stop_gradients=X)
  
    x = np.reshape(src_feat[q,:], (1,1,-1))
    y = np.reshape(tar_feat[q,:], (1,1,-1))
    p_op = np.reshape(mom_pitch[q:q+1,:], (1,1,-1))
    k = np.asarray([[6,50]])
    with tf.Session() as sess:
        w, p, iw, ip = sess.run([warped_curve, final_momenta, intermed_warped, intermed_momenta], 
                             feed_dict={X:x, P:p_op, K:k})
    
    x = np.reshape(x, (-1,1))
    y = np.reshape(y, (-1,1))
    w = np.reshape(w, (-1,1))
    iw = np.asarray(iw, np.float64)
    ip = np.asarray(ip, np.float64)
    p = np.asarray(p, np.float64)

#    pylab.clf()
#    pylab.plot(x, label="Source")
#    pylab.plot(y, label="Target")
#    pylab.plot(w, label="Warped")
#    pylab.legend()
    
    line_width = 3.5
    legend_size = 22
    figsize = (13, 8)
    pylab.rc('axes', linewidth=2)
    pylab.rc('font', size=20, weight='bold')

    pylab.figure(figsize=figsize)
    pylab.plot(np.arange(0,81,1), x.reshape((-1,1)), linewidth=line_width, color='b', label='Source F0')
    pylab.plot(np.arange(0,81,1), y.reshape((-1,1)), linewidth=line_width, color='g', label='Target F0')
    pylab.quiver(np.arange(0,81,1), x.reshape((-1,)), np.zeros((81,)), p_op.reshape((-1,)), 
                 width=0.005, color='k', scale=100, label='Initial Momenta')
    pylab.ylim([70, 300]), pylab.legend(loc=1, prop={'size': legend_size, 'weight': 'bold'})
    
    pylab.figure(figsize=figsize)
    pylab.plot(np.arange(0,81,1), x.reshape((-1,1)), linewidth=line_width, color='b', label='Source F0')
    pylab.plot(np.arange(0,81,1), y.reshape((-1,1)), linewidth=line_width, color='g', label='Target F0')
    pylab.plot(np.arange(0,81,1), iw.reshape((-1,1)), linewidth=line_width, color='m', label='Intermediate F0')
    pylab.quiver(np.arange(0,81,1), iw.reshape((-1,)), np.zeros((81,)), ip.reshape((-1,)), 
                 width=0.005, color='k', scale=180, label='Intermediate Momenta')
    pylab.ylim([70, 300]), pylab.legend(loc=1, prop={'size': legend_size, 'weight': 'bold'})
    
    pylab.figure(figsize=figsize)
    pylab.plot(np.arange(0,81,1), x.reshape((-1,1)), linewidth=line_width, color='b', label='Source F0')
    pylab.plot(np.arange(0,81,1), y.reshape((-1,1)), linewidth=line_width, color='g', label='Target F0')
    pylab.plot(np.arange(0,81,1), w.reshape((-1,1)), linewidth=line_width, color='m', label='Final F0')
    pylab.quiver(np.arange(0,81,1), w.reshape((-1,)), np.zeros((81,)), p.reshape((-1,)), 
                 width=0.005, color='k', scale=300, label='Final Momenta')
    pylab.ylim([70, 300]), pylab.legend(loc=1, prop={'size': legend_size, 'weight': 'bold'})    
    
    
    
    























