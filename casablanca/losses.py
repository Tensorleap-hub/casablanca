import tensorflow as tf


def zero_loss(y_true, y_pred):
    return tf.constant(0.0, dtype=tf.float32)
