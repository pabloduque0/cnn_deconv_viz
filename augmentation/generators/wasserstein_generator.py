from keras import models
from keras import layers
from augmentation import metrics
from keras.optimizers import Adam
import keras.backend as K
from keras import losses
import numpy as np

def create_model(input_shape, sub_start_shape=(6, 6, 128)):

    input_layer = layers.Input(shape=input_shape)

    conv_2 = layers.Dense(1024)(input_layer)
    lr_1 = layers.LeakyReLU()(conv_2)
    dense_2 = layers.Dense(np.prod(list(sub_start_shape)))(lr_1)
    bn_1 = layers.BatchNormalization()(dense_2)
    lr_2 = layers.LeakyReLU()(bn_1)
    print("prior shape: ", K.int_shape(lr_2))
    reshaped = layers.Reshape(sub_start_shape, input_shape=(np.prod(list(sub_start_shape)),))(lr_2)
    bn_axis = -1
    conv_1 = layers.Conv2DTranspose(128, (5, 5), strides=2, padding='same')(reshaped)
    bn_2 = layers.BatchNormalization(axis=bn_axis)(conv_1)
    lr_3 = layers.LeakyReLU()(bn_2)
    conv_2 = layers.Convolution2D(64, (5, 5), padding='same')(lr_3)
    bn_3 = layers.BatchNormalization(axis=bn_axis)(conv_2)
    lr_4 = layers.LeakyReLU()(bn_3)
    conv_3 = layers.Conv2DTranspose(64, (5, 5), strides=2, padding='same')(lr_4)
    bn_5 = layers.BatchNormalization(axis=bn_axis)(conv_3)
    lr_6 = layers.LeakyReLU()(bn_5)
    conv_4 = layers.Convolution2D(2, (5, 5), padding='same', activation='tanh')(lr_6)

    model = models.Model(inputs=input_layer, outputs=conv_4)
    model.summary()
    return model

