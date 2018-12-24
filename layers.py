import tensorflow as tf


def conv2d(x, out_channels, kernel_size=4, strides=2, padding='same', use_bias=False, name='Conv'):
    with tf.variable_scope(name):
        initializer = tf.random_normal_initializer(0, 0.02)
        return tf.layers.conv2d(x, out_channels, kernel_size=kernel_size, strides=strides, padding=padding,
                                kernel_initializer=initializer, use_bias=use_bias)


def deconv2d(x, out_channels, kernel_size=4, stride=2, padding='same', name="Deconv"):
    with tf.variable_scope(name):
        initializer = tf.random_normal_initializer(0, 0.02)
        return tf.layers.conv2d_transpose(x, out_channels, kernel_size=kernel_size, strides=stride, padding=padding,
                                          kernel_initializer=initializer, use_bias=False)


def res_block(x_in, out_channels=512, name='ResBlock'):
    with tf.variable_scope(name):
        x = conv2d(x_in, out_channels=out_channels, kernel_size=3, strides=1, name='Conv1')
        x = instance_norm(x, name='InstNorm1')
        x = tf.nn.relu(x, name='ReLU1')

        x = conv2d(x, out_channels=out_channels, kernel_size=3, strides=1, name='Conv2')
        x = instance_norm(x, name='InstNorm2')
        return tf.add(x_in, x)


def res_block_rfpad(x_in, out_channels=512, name='ResBlock'):
    with tf.variable_scope(name):
        x = reflect_pad(x_in, padsize=1, name='ReflectPad1')
        x = conv2d(x, out_channels=out_channels, kernel_size=3, strides=1, padding='valid', name='Conv1')
        x = instance_norm(x, name='InstNorm1')
        x = tf.nn.relu(x, name='ReLU1')

        x = reflect_pad(x, padsize=1, name='ReflectPad1')
        x = conv2d(x, out_channels=out_channels, kernel_size=3, strides=1, padding='valid', name='Conv2')
        x = instance_norm(x, name='InstNorm2')
        return tf.add(x_in, x)
        
        
def instance_norm(x, epsilon=1e-5, name='InstNorm'):
    with tf.variable_scope(name):
        stat_shape = x.get_shape().as_list()
        scale = tf.get_variable('scale', initializer=tf.random_normal_initializer(1.0, 0.02))
        shift = tf.get_variable('shift', initializer=tf.constant_initializer(0.0))

        inst_means, inst_vars = tf.nn.moments(x, axes=[1, 2], keep_dims=True)
        # Normalization
        inputs_normed = (x - inst_means) / tf.sqrt(inst_vars + epsilon)
        # Perform trainable shift.
        output = scale * inputs_normed + shift
        return output


def reflect_pad(x, padsize=1, name='ReflectPad'):
    with tf.variable_scope(name):
        h = tf.pad(x, paddings=[[0, 0], [padsize, padsize], [padsize, padsize], [0, 0]], mode='REFLECT')
        return h
