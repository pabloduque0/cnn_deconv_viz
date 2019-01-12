from keras import backend as K
import tensorflow as tf
import numpy as np

def custom_dice_coefficient(y_true, y_pred, conn_comp_weight=0.5):
    regular_dice = dice_coefficient(y_true, y_pred)
    conn_comp_true = tf.contrib.image.connected_components(tf.cast(tf.squeeze(y_true, axis=[-1]), tf.bool))
    conn_comp_pred = tf.contrib.image.connected_components(tf.cast(tf.squeeze(y_pred, axis=[-1]), tf.bool))

    n_conn_comp_true, _ = tf.unique(K.flatten(conn_comp_true))
    n_conn_comp_pred, _ = tf.unique(K.flatten(conn_comp_pred))
    conn_comp_ratio = tf.math.abs(tf.size(n_conn_comp_pred) - tf.size(n_conn_comp_true)) / tf.size(n_conn_comp_true)
    conn_comp_ratio = tf.cast(conn_comp_ratio, tf.float32)
    return regular_dice + ((1 - conn_comp_ratio) * regular_dice * conn_comp_weight)


def custom_dice_coefficient_loss(y_true, y_pred):
    return -custom_dice_coefficient(y_true, y_pred)


def dice_coefficient(y_true, y_pred, smooth=0.1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)

    intersection = K.sum(y_pred_f * y_true_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coefficient_loss(y_true, y_pred):
    return -dice_coefficient(y_true, y_pred)

def sigmoid(x):
    return 1. / (1. + K.exp(-x))

def segmentation_recall(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)

    recall = K.sum(y_pred_f * y_true_f) / tf.cast(K.sum(y_true_f), tf.float32)
    return recall


def weighted_crossentropy_pixelwise(y_true, y_pred):

    y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
    y_pred = K.log(y_pred / (1 - y_pred))

    wmh_indexes = np.where(y_true == 1.0)
    weights = np.repeat(1.0, 240 * 240)
    weights = np.reshape(weights, (1, 240, 240, 1))
    weights[wmh_indexes] = 5000.0

    crossentropy = (y_true * weights * -K.log(sigmoid(y_pred)) + (1 - y_true * weights) * -K.log(1 - sigmoid(y_pred)))
    return crossentropy


def prediction_count(y_true, y_pred):
    return tf.count_nonzero(y_pred)


def label_count(y_true, y_pred):
    return tf.count_nonzero(y_true)


def prediction_sum(y_true, y_pred):
    return tf.reduce_sum(y_pred)


def label_sum(y_true, y_pred):
    return tf.reduce_sum(y_true)


custom_dice_coef = custom_dice_coefficient
custom_dice_loss = custom_dice_coefficient_loss
dice_coef = dice_coefficient
dice_coef_loss = dice_coefficient_loss

weighted_crossentropy = weighted_crossentropy_pixelwise

predicted_count = prediction_count
predicted_sum = prediction_sum

ground_truth_count = label_count
ground_truth_sum =   label_sum

recall = segmentation_recall