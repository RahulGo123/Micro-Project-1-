import tensorflow as tf


def custom_huber_loss(y_true, y_pred):
    threshold = 1.0
    error = y_true - y_pred
    is_small_error = tf.abs(error) < threshold
    squared_error = tf.square(error) / 2
    linear_loss = threshold * tf.abs(error) - threshold**2 / 2
    return tf.where(is_small_error, squared_error, linear_loss)
