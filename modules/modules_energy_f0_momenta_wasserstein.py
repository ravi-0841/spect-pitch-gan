import tensorflow as tf 
from modules.base_modules_default_init import *


def sampler_pitch(input_pitch, input_mfc, final_filters=1, reuse=False, \
                       scope_name='sampler_pitch'):

    # Inputs have shape [batch_size, num_features, time]
    inputs = tf.concat([input_mfc, input_pitch], axis=1, \
                            name='sampler_input')
    
    # Cnvert it to [batch_size, time, num_features] for 1D convolution
    inputs_tranposed = tf.transpose(inputs, perm = [0, 2, 1], \
                            name='sampler_input_transpose')

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
#        r3 = residual1d_block(inputs=r2, filters=512, \
#                kernel_size=3, strides=1, \
#                name_prefix='residual1d_block3_')

        # Upsample
        u1 = upsample1d_block(inputs=r2, filters=512, \
                kernel_size=5, strides=1, \
                shuffle_size=2, name_prefix='upsample1d_block1_')
        u2 = upsample1d_block(inputs=u1, filters=256, \
                kernel_size=5, strides=1, \
                shuffle_size=2, name_prefix='upsample1d_block2_')
        
        # Dropout for stochasticity
        u2 = tf.nn.dropout(u2, keep_prob=0.5)

        # Output
        o1 = conv1d_layer(inputs=u2, filters=final_filters, \
                kernel_size=15, strides=1, \
                activation=None, name='o1_conv')

        o2 = tf.transpose(o1, perm=[0, 2, 1], name='output_transpose')
        return o2


def sampler_energy(input_pitch, input_mfc, final_filters=1, reuse=False, \
                       scope_name='sampler_energy'):

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
#        r3 = residual1d_block(inputs=r2, filters=512, \
#                kernel_size=3, strides=1, \
#                name_prefix='residual1d_block3_')

        # Upsample
        u1 = upsample1d_block(inputs=r2, filters=512, \
                kernel_size=5, strides=1, \
                shuffle_size=2, name_prefix='upsample1d_block1_')
        u2 = upsample1d_block(inputs=u1, filters=256, \
                kernel_size=5, strides=1, \
                shuffle_size=2, name_prefix='upsample1d_block2_')
        
        # Dropout for stochasticity
        u2 = tf.nn.dropout(u2, keep_prob=0.5)

        # Output
        o1 = conv1d_layer(inputs=u2, filters=final_filters, \
                kernel_size=15, strides=1, \
                activation=None, name='o1_conv')

        o2 = tf.transpose(o1, perm=[0, 2, 1], name='output_transpose')
        return o2
    

def discriminator(input_pitch, input_energy, 
        reuse=False, scope_name='discriminator'):

    # input_mfc and input_pitch has shape [batch_size, num_features, time]
    input_energy = tf.transpose(input_energy, perm=[0,2,1], 
            name='discriminator_energy_transpose')
    input_pitch = tf.transpose(input_pitch, perm=[0,2,1], 
            name='discriminator_pitch_transpose')

    with tf.variable_scope(scope_name) as scope:
        # Discriminator would be reused in CycleGAN
        if reuse:
            scope.reuse_variables()
        else:
            assert scope.reuse is False

        h1_energy = conv1d_layer(inputs=input_energy, filters=64, 
                kernel_size=3, strides=1, 
                activation=None, name='h1_energy_conv')
        h1_energy_gates = conv1d_layer(inputs=input_energy, filters=64, 
                kernel_size=3, strides=1, 
                activation=None, name='h1_energy_conv_gates')
        h1_energy_glu = gated_linear_layer(inputs=h1_energy, 
                gates=h1_energy_gates, name='h1_energy_glu')

        h1_pitch = conv1d_layer(inputs=input_pitch, filters=64, 
                kernel_size=3, strides=1, 
                activation=None, name='h1_pitch_conv')
        h1_pitch_gates = conv1d_layer(inputs=input_pitch, filters=64, 
                kernel_size=3, strides=1, 
                activation=None, name='h1_pitch_conv_gates')
        h1_pitch_glu = gated_linear_layer(inputs=h1_pitch, 
                gates=h1_pitch_gates, name='h1_pitch_glu')


        h1_glu = tf.concat([h1_energy_glu, h1_pitch_glu], axis=-1, 
                name='concat_inputs')
        
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
                             activation=None)

        return o1
