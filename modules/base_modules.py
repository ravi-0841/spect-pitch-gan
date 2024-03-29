import tensorflow as tf 

def gated_linear_layer(inputs, gates, name = None):

    activation = tf.multiply(x = inputs, \
                             y = tf.sigmoid(gates), name = name)

    return activation

def instance_norm_layer(
    inputs, 
    epsilon = 1e-06, 
    activation_fn = None, 
    name = None):

    instance_norm_layer = tf.contrib.layers.instance_norm(
        inputs = inputs,
        epsilon = epsilon,
        activation_fn = activation_fn)

    return instance_norm_layer

def conv1d_layer(
    inputs, 
    filters, 
    kernel_size, 
    strides = 1, 
    padding = 'same', 
    activation = None,
    kernel_initializer = tf.random_normal_initializer(mean=0.0, \
                            stddev=0.01, dtype=tf.float32),
    name = None): #0.01

    conv_layer = tf.layers.conv1d(
        inputs = inputs,
        filters = filters,
        kernel_size = kernel_size,
        strides = strides,
        padding = padding,
        activation = activation,
        kernel_initializer = kernel_initializer,
        name = name)

    return conv_layer

def conv2d_layer(
    inputs, 
    filters, 
    kernel_size, 
    strides, 
    padding = 'same', 
    activation = None,
    kernel_initializer = tf.random_normal_initializer(mean=0.0, \
                            stddev=0.01, dtype=tf.float32),
    name = None): #0.01

    conv_layer = tf.layers.conv2d(
        inputs = inputs,
        filters = filters,
        kernel_size = kernel_size,
        strides = strides,
        padding = padding,
        activation = activation,
        kernel_initializer = kernel_initializer,
        name = name)

    return conv_layer

def residual1d_block(
    inputs, 
    filters = 1024, 
    kernel_size = 3, 
    strides = 1,
    name_prefix = 'residual_block_'):

    h1 = conv1d_layer(inputs = inputs, filters = filters, \
                    kernel_size = kernel_size, strides = strides, \
                    activation = None, name = name_prefix + 'h1_conv')
    h1_norm = instance_norm_layer(inputs = h1, activation_fn = None, \
                    name = name_prefix + 'h1_norm')
    h1_gates = conv1d_layer(inputs = inputs, filters = filters, \
                    kernel_size = kernel_size, strides = strides, \
                    activation = None, name = name_prefix + 'h1_gates')
    h1_norm_gates = instance_norm_layer(inputs = h1_gates, \
                    activation_fn = None, \
                    name = name_prefix + 'h1_norm_gates')
    h1_glu = gated_linear_layer(inputs = h1_norm, \
                    gates = h1_norm_gates, \
                    name = name_prefix + 'h1_glu')
    h2 = conv1d_layer(inputs = h1_glu, filters = filters // 2, \
                    kernel_size = kernel_size, strides = strides, \
                    activation = None, name = name_prefix + 'h2_conv')
    h2_norm = instance_norm_layer(inputs = h2, \
                    activation_fn = None, name = name_prefix + 'h2_norm')
    
    h3 = inputs + h2_norm

    return h3

def downsample1d_block(
    inputs, 
    filters, 
    kernel_size, 
    strides,
    name_prefix = 'downsample1d_block_'):

    h1 = conv1d_layer(inputs = inputs, filters = filters, \
                kernel_size = kernel_size, strides = strides, \
                activation = None, name = name_prefix + 'h1_conv')
    h1_norm = instance_norm_layer(inputs = h1, \
                activation_fn = None, name = name_prefix + 'h1_norm')
    h1_gates = conv1d_layer(inputs = inputs, filters = filters, \
                kernel_size = kernel_size, strides = strides, \
                activation = None, name = name_prefix + 'h1_gates')
    h1_norm_gates = instance_norm_layer(inputs = h1_gates, \
                activation_fn = None, name = name_prefix + 'h1_norm_gates')
    h1_glu = gated_linear_layer(inputs = h1_norm, \
                gates = h1_norm_gates, name = name_prefix + 'h1_glu')

    return h1_glu

def downsample2d_block(
    inputs, 
    filters, 
    kernel_size, 
    strides,
    name_prefix = 'downsample2d_block_'):

    h1 = conv2d_layer(inputs = inputs, filters = filters, \
                kernel_size = kernel_size, strides = strides, \
                activation = None, name = name_prefix + 'h1_conv')
    h1_norm = instance_norm_layer(inputs = h1, \
                activation_fn = None, name = name_prefix + 'h1_norm')
    h1_gates = conv2d_layer(inputs = inputs, filters = filters, \
                kernel_size = kernel_size, strides = strides, \
                activation = None, name = name_prefix + 'h1_gates')
    h1_norm_gates = instance_norm_layer(inputs = h1_gates, \
                activation_fn = None, name = name_prefix + 'h1_norm_gates')
    h1_glu = gated_linear_layer(inputs = h1_norm, \
                gates = h1_norm_gates, name = name_prefix + 'h1_glu')

    return h1_glu

def upsample1d_block(
    inputs, 
    filters, 
    kernel_size, 
    strides,
    shuffle_size=2,
    name_prefix='upsample1d_block_'):
    
    h1 = conv1d_layer(inputs=inputs, filters=filters, \
                kernel_size=kernel_size, strides=strides, \
                activation=None, name=name_prefix + 'h1_conv')
    h1_shuffle = pixel_shuffler(inputs=h1, \
                shuffle_size=shuffle_size, \
                name=name_prefix + 'h1_shuffle')
    h1_norm = instance_norm_layer(inputs=h1_shuffle, \
                activation_fn=None, name=name_prefix + 'h1_norm')

    h1_gates = conv1d_layer(inputs=inputs, filters=filters, \
                kernel_size=kernel_size, strides=strides, \
                activation=None, name=name_prefix + 'h1_gates')
    h1_shuffle_gates = pixel_shuffler(inputs=h1_gates, \
                shuffle_size=shuffle_size, \
                name=name_prefix + 'h1_shuffle_gates')
    h1_norm_gates = instance_norm_layer(inputs=h1_shuffle_gates, \
                activation_fn=None, name=name_prefix + 'h1_norm_gates')

    h1_glu = gated_linear_layer(inputs=h1_norm, \
                gates=h1_norm_gates, name=name_prefix + 'h1_glu')

    return h1_glu

def pixel_shuffler(inputs, shuffle_size=2, name=None):

    n = tf.shape(inputs)[0]
    w = tf.shape(inputs)[1]
    c = inputs.get_shape().as_list()[2]

    oc = c // shuffle_size
    ow = w * shuffle_size

    outputs = tf.reshape(tensor = inputs, shape = [n, ow, oc], name = name)

    return outputs

def adaptive_normalization(inputs, gamma_statistics, beta_statistics, name=None):
    """
    Adaptive normalization gamma*input + beta
    """
    return None

def create_filter_mask(center_frequencies, filter_size=15, num_masks=64, 
        name_prefix='create_mask'):

    masks = []
    y = tf.range(start=0, limit=513, dtype=tf.float32, name=name_prefix+'_range')
    y = tf.reshape(y, [1, 513], name=name_prefix+'_range_reshape')
    for i in range(num_masks):
        center_frequency = center_frequencies[0, i]
        bin_center = tf.cast(7 + center_frequency*498, dtype=tf.int32, 
                name=name_prefix+'_bin_center_%d'%i)
        y_gauss = tf.divide(tf.exp(-1 * (y - bin_center)**2 / (2*6.49)), 0.157, 
                name=name_prefix+'_pdf_%d'%i)
        y_gauss = tf.matmul(tf.ones([filter_size, 1], dtype=tf.float32), y_gauss, 
                name=name_prefix+'_repmat_%d'%i)
        y_gauss_dct = tf.signal.dct(y_gauss, name=name_prefix+'_dct_%d'%i)
        y_gauss_dct = tf.divide(y_gauss_dct[:,:23], tf.math.sqrt(1024), 
                name=name_prefix+'_normalize_%d'%i)
        masks.append(y_gauss_dct)

    return masks

def apply_filter_mask(input, filter_mask, name_prefix='apply_mask'):
    return None
