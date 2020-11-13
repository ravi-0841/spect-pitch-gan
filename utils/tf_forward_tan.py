import tensorflow as tf


def dHr_tan(x, p, Css, kernel, name_prefix='compute_dHr'):
    """
    Computes the small displacement of x and p
    """
#    input_shape = x.get_shape().as_list()
    n = tf.cast(tf.shape(x)[1], dtype=tf.float32, name=name_prefix+'_shape')#input_shape[1]
    
    x_repeated_cols = tf.einsum('ijk,kl->ijl', x, tf.ones([1,n], 
        dtype=tf.float32), name=name_prefix+'_repeat_curve_data_cols')
    x_repeated_rows = tf.einsum('lj,ijk->ilk',tf.ones([n,1],dtype=tf.float32), \
                                tf.transpose(x, perm=[0,2,1]), 
                                name=name_prefix+'_repeat_curve_data_rows')
    
    x_repeated_diff = tf.subtract(x_repeated_cols, x_repeated_rows, 
            name=name_prefix+'_repeat_difference')
    S = tf.add(Css, tf.divide(tf.square(x_repeated_diff), 
        kernel[0,1]**2), name=name_prefix+'_compute_S_matrix')
    A = tf.exp(tf.multiply(-1.0,S), name=name_prefix+'_exponentiate_S')
    B = tf.divide(tf.multiply(-1.0,A), kernel[0,1]**2, 
            name=name_prefix+'_divide_by_kernel')
    dxHr = tf.einsum('ijk,ikl->ijl', A, p, name=name_prefix+'_compute_dxHr')
    
    C = x_repeated_diff
    prod_BC = tf.multiply(B, C, name=name_prefix+'_multiply_B_C')
    prod_PP = tf.einsum('ijk,ikl->ijl', p, 
            tf.transpose(p, perm=[0,2,1]), name=name_prefix+'_momenta_momenta_product')
    dpHr = tf.multiply(2.0, tf.reduce_sum(tf.multiply(prod_BC, prod_PP), axis=2, \
                               keepdims=True), name=name_prefix+'_compute_dpHr')
    return dxHr, dpHr


def fdh(x, p, Css, h, kernel, name_prefix='forward_dh'):
    """
    compute displacement for step size h
    """
    dx, dp = dHr_tan(x, p, Css, kernel, name_prefix=name_prefix+'_dHr')
    kx = tf.add(x, tf.multiply(h, dx), name=name_prefix+'_add_dx')
    kp = tf.subtract(p, tf.multiply(h, dp), name=name_prefix+'_subtract_dp')
    return kx, kp


def lddmm(x, p, kernel, num_iter=3, reuse=False, scope_name='warping_generator'):
    """
    Expecting input x to be of shape [#batch, 1, #length]
    Expecting momenta p to be of the shape[#batch, 1, #length]
    """
    x = tf.transpose(x, perm=[0,2,1], name='input_curve_transpose') # converts x to shape[#batch, #length, 1]
    p = tf.transpose(p, perm=[0,2,1], name='input_momena_transpose') # converts p to shape[#batch, #length, 1]

    with tf.variable_scope(scope_name) as scope:
        if reuse:
            scope.reuse_variables()
        else:
            assert scope.reuse is False

        input_shape = tf.cast(tf.shape(x), dtype=tf.float32, name='input_shape_casting')
        time_axis = tf.expand_dims(tf.range(start=0, limit=input_shape[1], 
            dtype=tf.float32), axis=-1, name='expand_time_axis')
        repeated_cols = tf.matmul(time_axis, 
                tf.ones_like(tf.transpose(time_axis,perm=[1,0])), name='repeating_time_value')
        repeated_rows = tf.matmul(tf.ones_like(time_axis), 
                tf.transpose(time_axis, perm=[1,0]), name='repeating_curve_values')
        Css = tf.divide(tf.square(repeated_cols -  repeated_rows), (kernel[0,0]**2), 
                name='square_diff_matrix')
        
        dt = 1.0 / num_iter #3.0
        x_evol = x
        p_evol = p
        
        x2, p2 = fdh(x_evol, p_evol, Css, dt/2, kernel, name_prefix='fdh_0_0')
        x3, p3 = fdh(x2, p2, Css, dt, kernel, name_prefix='fdh_0_1')
        
        #-----------------------------------------------------------------------------------------------------
        """
        Iterating over n time steps
        """

        for i in range(num_iter-1):
            x_evol = tf.add(tf.subtract(x3,x2),x_evol, name='add_xevol_'+str(i+1))
            p_evol = tf.add(tf.subtract(p3,p2),p_evol, name='add_pevol_'+str(i+1))
            
            x2, p2 = fdh(x_evol, p_evol, Css, dt/2, kernel, name_prefix='fdh_%i_0'%(i+1))
            x3, p3 = fdh(x2, p2, Css, dt, kernel, name_prefix='fdh_%i_1'%(i+1))
            
#            x_evol = tf.add(tf.subtract(x3,x2),x_evol)
#            p_evol = tf.add(tf.subtract(p3,p2),p_evol)
#            
#            x2, p2 = fdh(x_evol, p_evol, Css, dt/2, kernel)
#            x3, p3 = fdh(x2, p2, Css, dt, kernel)
        #------------------------------------------------------------------------------------------------------

        output = tf.add(tf.subtract(x3, x2), x_evol, name='final_x_evol_addition')
        return tf.transpose(output, perm=[0,2,1])


#if __name__ == "__main__":
#    tf.reset_default_graph()
#    import scipy.io as scio
#    import numpy as np
#    import pylab
#    data = scio.loadmat('/home/ravi/Desktop/momentum-warping/data/neu-ang/mom-valid.mat')
#    src_feat = np.asarray(data['src_f0_feat'], np.float64)
#    tar_feat = np.asarray(data['tar_f0_feat'], np.float64)
#    mom_pitch = np.asarray(data['momentum_pitch'], np.float64)
#    src_feat[np.where(src_feat<=0)] = 1e-1
#    tar_feat[np.where(tar_feat<=0)] = 1e-1
#    q = np.random.randint(0, src_feat.shape[0])
#    
#    X = tf.placeholder(dtype=tf.float32, shape=[None, 1, None], name="input_curve")
#    P = tf.placeholder(dtype=tf.float32, shape=[None, 1, None], name="momenta")
#    K = tf.placeholder(dtype=tf.float32, shape=[1,2], name="kernel")
#    
#    warped_curve = lddmm(X, P, K, num_iter=5)
#    grads = tf.gradients(ys=warped_curve, xs=P, stop_gradients=X)
#  
#    x = np.reshape(src_feat[q,:], (1,1,-1))
#    y = np.reshape(tar_feat[q,:], (1,1,-1))
#    p_op = np.reshape(mom_pitch[q:q+1,:], (1,1,-1))
#    k = np.asarray([[6,50]])
#    with tf.Session() as sess:
#        w,g = sess.run([warped_curve, grads], feed_dict={X:x, P:p_op, K:k})
#    
#    x = np.reshape(x, (-1,1))
#    y = np.reshape(y, (-1,1))
#    w = np.reshape(w, (-1,1))
#
#    pylab.clf()
#    pylab.plot(x, label="Source")
#    pylab.plot(y, label="Target")
#    pylab.plot(w, label="Warped")
#    pylab.legend()
    

    
    
    























