from keras import models
from keras import layers
from contextlib import redirect_stdout
import os
from keras.callbacks import ModelCheckpoint, TensorBoard
from sklearn.model_selection import train_test_split
from keras.models import load_model
from keras.optimizers import Adam, SGD
from metrics import dice_coef, dice_coef_loss, weighted_crossentropy, predicted_count, ground_truth_count, \
    ground_truth_sum, predicted_sum, recall, custom_dice_coef, custom_dice_loss
from keras.losses import binary_crossentropy
import cv2
import numpy as np
from basenetwork import BaseNetwork

class OldUnet(BaseNetwork):

    def __init__(self, model_path=None, img_shape=None):

        if model_path is None:
            if img_shape is None:
                raise Exception('If no model path is provided img shape is a mandatory argument.')
            model = self.create_model(img_shape)
        else:
            model = load_model(model_path)

        super().__init__(model)


    def create_model(self, img_shape):

        concat_axis = 3

        inputs = layers.Input(shape=img_shape)
        conv1 = layers.Conv2D(64, kernel_size=5, padding='same', kernel_initializer='he_normal', activation='relu')(inputs)
        conv2 = layers.Conv2D(64, kernel_size=5, padding='same', kernel_initializer='he_normal', activation='relu')(conv1)
        maxpool1 = layers.MaxPool2D(pool_size=(2, 2))(conv2)

        conv3 = layers.Conv2D(128, kernel_size=5, padding='same', kernel_initializer='he_normal', activation='relu')(maxpool1)
        conv4 = layers.Conv2D(128, kernel_size=5, padding='same', kernel_initializer='he_normal', activation='relu')(conv3)
        maxpool2 = layers.MaxPool2D(pool_size=(2, 2))(conv4)

        conv5 = layers.Conv2D(256, kernel_size=5, padding='same', kernel_initializer='he_normal', activation='relu')(maxpool2)
        conv6 = layers.Conv2D(256, kernel_size=5, padding='same', kernel_initializer='he_normal', activation='relu')(conv5)
        maxpool3 = layers.MaxPool2D(pool_size=(2, 2))(conv6)

        conv7 = layers.Conv2D(512, kernel_size=5, padding='same', kernel_initializer='he_normal', activation='relu')(maxpool3)
        conv8 = layers.Conv2D(512, kernel_size=5, padding='same', kernel_initializer='he_normal', activation='relu')(conv7)
        maxpool4 = layers.MaxPool2D(pool_size=(2, 2))(conv8)

        conv9 = layers.Conv2D(1024, kernel_size=5, padding='same', kernel_initializer='he_normal', activation='relu')(maxpool4)
        conv10 = layers.Conv2D(1024, kernel_size=5, padding='same', kernel_initializer='he_normal', activation='relu')(conv9)

        up_conv10 = layers.UpSampling2D(size=(2, 2))(conv10)
        up_samp1 = layers.concatenate([conv8, up_conv10], axis=concat_axis)
        conv11 = layers.Conv2D(512, kernel_size=5, padding='same', kernel_initializer='he_normal', activation='relu')(up_samp1)

        conv12 = layers.Conv2D(512, kernel_size=5, padding='same', kernel_initializer='he_normal', activation='relu')(conv11)
        conv13 = layers.Conv2D(512, kernel_size=5, padding='same', kernel_initializer='he_normal', activation='relu')(conv12)

        up_conv13 = layers.UpSampling2D(size=(2, 2))(conv13)
        up_samp2 = layers.concatenate([conv6, up_conv13], axis=concat_axis)
        conv14 = layers.Conv2D(256, kernel_size=5, padding='same', kernel_initializer='he_normal', activation='relu')(up_samp2)

        conv15 = layers.Conv2D(256, kernel_size=5, padding='same', kernel_initializer='he_normal', activation='relu')(conv14)
        conv16 = layers.Conv2D(256, kernel_size=5, padding='same', kernel_initializer='he_normal', activation='relu')(conv15)

        up_conv16 = layers.UpSampling2D(size=(2, 2))(conv16)
        up_samp3 = layers.concatenate([conv4, up_conv16], axis=concat_axis)
        conv17 = layers.Conv2D(128, kernel_size=5, padding='same', kernel_initializer='he_normal', activation='relu')(up_samp3)

        conv18 = layers.Conv2D(128, kernel_size=5, padding='same', kernel_initializer='he_normal', activation='relu')(conv17)
        conv19 = layers.Conv2D(128, kernel_size=5, padding='same', kernel_initializer='he_normal', activation='relu')(conv18)

        up_conv19 = layers.UpSampling2D(size=(2, 2))(conv19)
        up_samp4 = layers.concatenate([conv2, up_conv19], axis=concat_axis)
        conv20 = layers.Conv2D(64, kernel_size=5, padding='same', kernel_initializer='he_normal', activation='relu')(up_samp4)

        conv21 = layers.Conv2D(64, kernel_size=5, padding='same', kernel_initializer='he_normal', activation='relu')(conv20)
        conv22 = layers.Conv2D(64, kernel_size=5, padding='same', kernel_initializer='he_normal', activation='relu')(conv21)

        conv23 = layers.Conv2D(1, kernel_size=1, padding='same', kernel_initializer='he_normal', activation='sigmoid')(conv22)

        model = models.Model(inputs=inputs, outputs=conv23)

        model.compile(optimizer=Adam(lr=0.000001), loss=dice_coef_loss, metrics=[dice_coef, binary_crossentropy, weighted_crossentropy,
                                                                                   predicted_count, predicted_sum, ground_truth_count,
                                                                                 ground_truth_sum, recall])

        model.summary()

        return model