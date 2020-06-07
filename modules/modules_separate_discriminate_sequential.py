import tensorflow as tf 
from modules.base_modules_default_init import *


def generator(input_pitch, input_mfc, final_filters=1, reuse=False, \
                       scope_name='generator'):

    # Inputs have shape [batch_size, num_features, time]
    inputs = tf.concat([input_mfc, input_pitch], axis=1, \
                            name='generator_input')
    
    # Cnvert it to [batch_size, time, num_features] for 1D convolution
    inputs_tranposed = tf.transpose(inputs, perm = [0, 2, 1], \
                            name='generator_input_transpose')

    with tf.variable_scope(scope_name) as scope:
        # Discriminator would be reused in CycleGAN
        if reuse:
            scope.reuse_variables()
        else:
            assert scope.reuse is False

        h1 = conv1d_layer(inputs=inputs_tranposed, filters=64, \
                kernel_size=15, strides=1, \
                activation=None, name='h1_conv')
        h1_gates = conv1d_layer(inputs=inputs_tranposed, filters=64, \
                kernel_size=15, strides=1, \
                activation=None, name='h1_conv_gates')
        h1_glu = gated_linear_layer(inputs=h1, \
                gates=h1_gates, name='h1_glu')
        
        # Downsample
        d1 = downsample1d_block(inputs=h1_glu, filters=128, \
                kernel_size=5, strides=2, \
                name_prefix='downsample1d_block1_')
        d2 = downsample1d_block(inputs=d1, filters=256, \
                kernel_size=5, strides=2, \
                name_prefix='downsample1d_block2_')

        # Residual blocks
        r1 = residual1d_block(inputs=d2, filters=512, \
                kernel_size=3, strides=1, \
                name_prefix='residual1d_block1_')
        r2 = residual1d_block(inputs=r1, filters=512, \
                kernel_size=3, strides=1, \
                name_prefix='residual1d_block2_')
        r3 = residual1d_block(inputs=r2, filters=512, \
                kernel_size=3, strides=1, \
                name_prefix='residual1d_block3_')

        # Upsample
        u1 = upsample1d_block(inputs=r3, filters=512, \
                kernel_size=5, strides=1, \
                shuffle_size=2, name_prefix='upsample1d_block1_')
        u2 = upsample1d_block(inputs=u1, filters=256, \
                kernel_size=5, strides=1, \
                shuffle_size=2, name_prefix='upsample1d_block2_')
        
        # Dropout for stochasticity
        u2 = tf.nn.dropout(u2, keep_prob=0.9)

        # Output
        o1 = conv1d_layer(inputs=u2, filters=final_filters, \
                kernel_size=15, strides=1, \
                activation=None, name='o1_conv')

#        o2 = conv1d_layer(inputs=o1, filters=1, \
#                kernel_size=15, strides=1, \
#                activation=None, name='o2_conv')

        o2 = tf.transpose(o1, perm=[0, 2, 1], name='output_transpose')
#        o3 = tf.reduce_mean(o2, axis=1, keepdims=True)
        return o2


def predictor(input_pitch, input_mfc, final_filters=23, reuse=False, \
                       scope_name='predictor'):

    # Inputs have shape [batch_size, num_features, time]
    inputs = tf.concat([input_mfc, input_pitch], axis=1, \
                            name='predictor_input')
    
    # Cnvert it to [batch_size, time, num_features] for 1D convolution
    inputs_tranposed = tf.transpose(inputs, perm = [0, 2, 1], \
                            name='predictor_input_transpose')

    with tf.variable_scope(scope_name) as scope:
        # Discriminator would be reused in CycleGAN
        if reuse:
            scope.reuse_variables()
        else:
            assert scope.reuse is False

        h1 = conv1d_layer(inputs=inputs_tranposed, filters=64, \
                kernel_size=15, strides=1, \
                activation=None, name='h1_conv')
        h1_gates = conv1d_layer(inputs=inputs_tranposed, filters=64, \
                kernel_size=15, strides=1, \
                activation=None, name='h1_conv_gates')
        h1_glu = gated_linear_layer(inputs=h1, \
                gates=h1_gates, name='h1_glu')
        
        # Downsample
        d1 = downsample1d_block(inputs=h1_glu, filters=128, \
                kernel_size=5, strides=2, \
                name_prefix='downsample1d_block1_')
        d2 = downsample1d_block(inputs=d1, filters=256, \
                kernel_size=5, strides=2, \
                name_prefix='downsample1d_block2_')

        # Residual blocks
        r1 = residual1d_block(inputs=d2, filters=512, \
                kernel_size=3, strides=1, \
                name_prefix='residual1d_block1_')
        r2 = residual1d_block(inputs=r1, filters=512, \
                kernel_size=3, strides=1, \
                name_prefix='residual1d_block2_')
        r3 = residual1d_block(inputs=r2, filters=512, \
                kernel_size=3, strides=1, \
                name_prefix='residual1d_block3_')
        r4 = residual1d_block(inputs=r3, filters=512, \
                kernel_size=3, strides=1, \
                name_prefix='residual1d_block4_')

        # Upsample
        u1 = upsample1d_block(inputs=r4, filters=512, \
                kernel_size=5, strides=1, \
                shuffle_size=2, name_prefix='upsample1d_block1_')
        u2 = upsample1d_block(inputs=u1, filters=256, \
                kernel_size=5, strides=1, \
                shuffle_size=2, name_prefix='upsample1d_block2_')
        
        # Dropout for stochasticity
        u2 = tf.nn.dropout(u2, keep_prob=1.0)

        # Output
        o1 = conv1d_layer(inputs=u2, filters=final_filters, \
                kernel_size=15, strides=1, \
                activation=None, name='o1_conv')

#        o2 = conv1d_layer(inputs=o1, filters=1, \
#                kernel_size=15, strides=1, \
#                activation=None, name='o2_conv')

        o2 = tf.transpose(o1, perm=[0, 2, 1], name='output_transpose')
#        o3 = tf.reduce_mean(o2, axis=1, keepdims=True)
        
        return o2
    

def discriminator(input_mfc_1, input_mfc_2, 
        reuse=False, scope_name='discriminator'):

    # input_mfc and input_pitch has shape [batch_size, num_features, time]
    input_mfc = tf.transpose(tf.concat([input_mfc_1, input_mfc_2], axis=1), 
                             perm=[0,2,1], name='discriminator_mfc_transpose')

    with tf.variable_scope(scope_name) as scope:
        # Discriminator would be reused in CycleGAN
        if reuse:
            scope.reuse_variables()
        else:
            assert scope.reuse is False

        h1 = conv1d_layer(inputs=input_mfc, filters=64, 
                kernel_size=3, strides=1, 
                activation=None, name='h1_conv')
        h1_gates = conv1d_layer(inputs=input_mfc, filters=64, 
                kernel_size=3, strides=1, 
                activation=None, name='h1_conv_gates')
        h1_glu = gated_linear_layer(inputs=h1, 
                gates=h1_gates, name='h1_glu')
        
        d1 = downsample1d_block(inputs=h1_glu, filters=128, 
                kernel_size=3, strides=2, 
                name_prefix='downsample2d_block1_')
        d2 = downsample1d_block(inputs=d1, filters=256, 
                kernel_size=3, strides=2, 
                name_prefix='downsample2d_block2_')
        d3 = downsample1d_block(inputs=d2, filters=256, 
                kernel_size=3, strides=2, 
                name_prefix='downsample2d_block3_')

        # Output
        o1 = tf.layers.dense(inputs=d3, units=1, \
                             activation=tf.nn.sigmoid)

        return o1
