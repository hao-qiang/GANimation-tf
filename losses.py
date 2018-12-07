import tensorflow as tf


def l1_loss(y_true, y_pred):
    return tf.reduce_mean(tf.abs(y_true - y_pred))


def l2_loss(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_true - y_pred))


def smooth_loss(mat):
    return tf.reduce_sum(tf.abs(mat[:, :, :-1, :] - mat[:, :, 1:, :])) + \
           tf.reduce_sum(tf.abs(mat[:, :-1, :, :] - mat[:, 1:, :, :]))
