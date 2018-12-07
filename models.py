import tensorflow as tf
from layers import conv2d, deconv2d, instance_norm, res_block


def generator(real_img, desired_au, reuse=False):
    '''
    :param:
        real_img: RGB face images, shape [batch, 128, 128, 3], value [-1,1].
        desired_au: AU value, shape [batch, 17], value [0,1].
    :return:
        fake_img: RGB generate face, shape [batch, 128, 128, 3], value [-1,1].
        fake_mask: face mask, shape [batch, 128, 128, 1], value [0,1].
    '''
    with tf.variable_scope('generator') as scope:
        if reuse:
            scope.reuse_variables()
            
        desired_au = tf.expand_dims(desired_au, axis=1, name='ExpandDims1')
        desired_au = tf.expand_dims(desired_au, axis=2, name='ExpandDims2')
        desired_au = tf.tile(desired_au, multiples=[1,128,128,1], name='Tile')
        x = tf.concat([real_img, desired_au], axis=3, name='Concat')
        
        x = conv2d(x, out_channels=64, kernel_size=7, strides=1, name='Conv1')
        x = instance_norm(x, name='InstNorm1')
        x = tf.nn.relu(x, name='ReLU1')

        x = conv2d(x, out_channels=128, kernel_size=4, strides=2, name='Conv2')
        x = instance_norm(x, name='InstNorm2')
        x = tf.nn.relu(x, name='ReLU2')

        x = conv2d(x, out_channels=256, kernel_size=4, strides=2, name='Conv3')
        x = instance_norm(x, name='InstNorm3')
        x = tf.nn.relu(x, name='ReLU3')

        for i in range(1, 7):
            x = res_block(x, out_channels=256, name='ResBlock'+str(i))

        x = deconv2d(x, out_channels=128, kernel_size=4, stride=2, name='Deconv1')
        x = instance_norm(x, name='InstNorm4')
        x = tf.nn.relu(x, name='ReLU4')

        x = deconv2d(x, out_channels=64, kernel_size=4, stride=2, name='Deconv2')
        x = instance_norm(x, name='InstNorm5')
        features = tf.nn.relu(x, name='ReLU5')

        x = conv2d(features, out_channels=3, kernel_size=7, strides=1, name='ConvImg')
        fake_img = tf.tanh(x, name='Tanh')

        x = conv2d(features, out_channels=1, kernel_size=7, strides=1, name='ConvMask')
        fake_mask = tf.sigmoid(x, name='Sigmoid')

        return fake_img, fake_mask


def discriminator(x, reuse=False):
    '''
    :param:
        x: RGB face images, shape [batch, 128, 128, 3], value [-1,1].
    :return:
        pred_img: shape [batch, 2, 2, 1].
        pred_au: AU prediction, shape [batch, 17], value [0,1].
    '''
    with tf.variable_scope('discriminator') as scope:
        if reuse:
            scope.reuse_variables()

        for i in range(6):
            x = conv2d(x, out_channels=64*(2**i), kernel_size=4, strides=2, use_bias=True, name='Conv'+str(i+1))
            x = tf.nn.leaky_relu(x, alpha=0.01, name='LReLU'+str(i+1))

        pred_img = conv2d(x, out_channels=1, kernel_size=3, strides=1, name='PredImg')
        pred_au = conv2d(x, out_channels=17, kernel_size=2, strides=1, padding='valid', name='ConvAU')
        pred_au = tf.squeeze(pred_au, [1,2], name='PredAU')
        return pred_img, pred_au
