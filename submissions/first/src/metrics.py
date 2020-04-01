from keras import backend as K
import tensorflow as tf
import numpy as np
import tensorflow_addons as tfa

def custom_dice_coefficient(y_true, y_pred, recall_weight=0.3):
    recall_weight = tf.Variable(recall_weight, dtype=tf.float32)
    regular_dice = dice_coefficient(y_true, y_pred)
    recall = lession_recall(y_true, y_pred)
    recall = tf.cast(recall, dtype=tf.float32)
    recall_addition = recall * regular_dice * recall_weight
    return regular_dice + recall_addition


def lession_recall(y_true, y_pred):
    conn_comp_true = tfa.image.connected_components(tf.cast(tf.squeeze(y_true, axis=[-1]), tf.bool))
    conn_comp_pred = conn_comp_true * tf.cast(tf.squeeze(y_pred, axis=[-1]), tf.int32)

    n_conn_comp_true, _ = tf.unique(K.flatten(conn_comp_true))
    n_conn_comp_pred, _ = tf.unique(K.flatten(conn_comp_pred))
    n_conn_comp_true = tf.size(input=n_conn_comp_true) - 1
    n_conn_comp_pred = tf.size(input=n_conn_comp_pred) - 1

    recall = tf.cond(pred=tf.equal(n_conn_comp_pred, tf.Variable(0)),
                     true_fn=lambda: tf.Variable(1.0, dtype=tf.float64), false_fn=lambda: n_conn_comp_pred / n_conn_comp_true)
    return recall


def thresholded_dice(y_true, y_pred):
    y_true = tf.math.floor(y_true + 0.6)
    return dice_coefficient(y_true, y_pred)

def thresholded_dice_loss(y_true, y_pred):
    return -thresholded_dice(y_true, y_pred)

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
    return tf.math.count_nonzero(y_pred)


def label_count(y_true, y_pred):
    return tf.math.count_nonzero(y_true)


def prediction_sum(y_true, y_pred):
    return tf.reduce_sum(input_tensor=y_pred)


def label_sum(y_true, y_pred):
    return tf.reduce_sum(input_tensor=y_true)


custom_dice_coef = custom_dice_coefficient
custom_dice_loss = custom_dice_coefficient_loss
dice_coef = dice_coefficient
dice_coef_loss = dice_coefficient_loss

weighted_crossentropy = weighted_crossentropy_pixelwise

predicted_count = prediction_count
predicted_sum = prediction_sum

ground_truth_count = label_count
ground_truth_sum =   label_sum

pixel_recall = segmentation_recall

obj_recall = lession_recall