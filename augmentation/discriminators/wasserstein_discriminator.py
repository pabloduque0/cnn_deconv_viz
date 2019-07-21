from keras import models
from keras import layers
import tensorflow as tf
from keras.optimizers import Adam
from augmentation import metrics
from keras import losses
import keras.backend as K


def create_model(input_shape):

    input_layer = layers.Input(shape=input_shape)

    conv_1 = layers.Convolution2D(64, (5, 5), padding='same', input_shape=input_shape)(input_layer)
    lr_1 = layers.LeakyReLU()(conv_1)
    conv_2 = layers.Convolution2D(128, (5, 5), kernel_initializer='he_normal', strides=[2, 2])(lr_1)
    lr_2 = layers.LeakyReLU()(conv_2)
    conv_3 = layers.Convolution2D(128, (5, 5), kernel_initializer='he_normal', padding='same', strides=[2, 2])(lr_2)
    lr_3 = layers.LeakyReLU()(conv_3)
    flatten = layers.Flatten()(lr_3)
    dense_1 = layers.Dense(1024, kernel_initializer='he_normal')(flatten)
    lr_4 = layers.LeakyReLU()(dense_1)
    dense_2 = layers.Dense(1, kernel_initializer='he_normal')(lr_4)

    model = models.Model(inputs=input_layer, outputs=dense_2)
    model.summary()

    return model
