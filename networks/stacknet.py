from keras import models
from keras import layers
import os
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.models import load_model
from keras.optimizers import Adam
from networks.metrics import dice_coef, dice_coef_loss, weighted_crossentropy, predicted_count, \
    ground_truth_count, ground_truth_sum, predicted_sum, pixel_recall, custom_dice_coef, custom_dice_loss, obj_recall
from keras.losses import binary_crossentropy
import cv2
import numpy as np
from networks import custom_layers
import keras.backend as K
from networks.basenetwork import BaseNetwork

class StackNet(BaseNetwork):

    def __init__(self,  model_path=None, img_shape=None):

        if model_path is None:
            if img_shape is None:
                raise Exception('If no model path is provided img shape is a mandatory argument.')
            model = self.create_model(img_shape)
        else:
            model = load_model(model_path)

        super().__init__(model, img_shape)


    def create_model(self, img_shape):
        concat_axis = 3

        inputs = layers.Input(shape=img_shape)

        stack1 = layers.Conv2D(40, kernel_size=5, padding='same', kernel_initializer='he_normal', activation='relu')(
            inputs)
        stack2 = layers.Conv2D(40, kernel_size=5, padding='same', kernel_initializer='he_normal', activation='relu')(
            stack1)
        stack3 = layers.Conv2D(40, kernel_size=5, padding='same', kernel_initializer='he_normal', activation='relu')(
            stack2)

        conv1 = layers.Conv2D(40, kernel_size=5, padding='same', kernel_initializer='he_normal', activation='relu')(
            stack3)
        conv2 = layers.Conv2D(40, kernel_size=5, padding='same', kernel_initializer='he_normal', activation='relu')(
            conv1)
        #conv2 = layers.BatchNormalization(axis=concat_axis, momentum=0.99, epsilon=0.001, center=True, scale=True)(conv2)

        maxpool1 = layers.MaxPool2D(pool_size=(2, 2))(conv2)

        conv3 = layers.Conv2D(80, kernel_size=5, padding='same', kernel_initializer='he_normal', activation='relu')(
            maxpool1)
        conv4 = layers.Conv2D(80, kernel_size=5, padding='same', kernel_initializer='he_normal', activation='relu')(
            conv3)
        #conv4 = layers.BatchNormalization(axis=concat_axis, momentum=0.99, epsilon=0.001, center=True, scale=True)(conv4)
        maxpool2 = layers.MaxPool2D(pool_size=(2, 2))(conv4)

        conv5 = layers.Conv2D(112, kernel_size=5, padding='same', kernel_initializer='he_normal', activation='relu')(
            maxpool2)
        conv6 = layers.Conv2D(112, kernel_size=5, padding='same', kernel_initializer='he_normal', activation='relu')(
            conv5)
        #conv6 = layers.BatchNormalization(axis=concat_axis, momentum=0.99, epsilon=0.001, center=True, scale=True)(conv6)
        maxpool3 = layers.MaxPool2D(pool_size=(2, 2))(conv6)

        conv7 = layers.Conv2D(224, kernel_size=5, padding='same', kernel_initializer='he_normal', activation='relu')(
            maxpool3)
        conv8 = layers.Conv2D(224, kernel_size=3, padding='same', kernel_initializer='he_normal', activation='relu')(
            conv7)
        #conv8 = layers.BatchNormalization(axis=concat_axis, momentum=0.99, epsilon=0.001, center=True, scale=True)(conv8)
        maxpool4 = layers.MaxPool2D(pool_size=(2, 2))(conv8)

        conv9 = layers.Conv2D(224, kernel_size=5, padding='same', kernel_initializer='he_normal', activation='relu')(
            maxpool4)
        conv10 = layers.Conv2D(416, kernel_size=5, padding='same', kernel_initializer='he_normal', activation='relu')(
            conv9)
        #conv10 = layers.BatchNormalization(axis=concat_axis, momentum=0.99, epsilon=0.001, center=True, scale=True)(conv10)
        up_conv10 = layers.UpSampling2D(size=(2, 2))(conv10)
        up_conv1 = layers.Conv2D(416, kernel_size=5, padding='same',
                                 kernel_initializer='he_normal', activation='relu')(up_conv10)
        ch, cw = self.get_crop_shape(conv8, up_conv1)
        crop_conv8 = layers.Cropping2D(cropping=(ch, cw))(conv8)
        up_samp1 = layers.concatenate([crop_conv8, up_conv1], axis=concat_axis)

        conv11 = layers.Conv2D(224, kernel_size=5, padding='same', kernel_initializer='he_normal', activation='relu')(
            up_samp1)
        conv12 = layers.Conv2D(224, kernel_size=5, padding='same', kernel_initializer='he_normal', activation='relu')(
            conv11)
        #conv12 = layers.BatchNormalization(axis=concat_axis, momentum=0.99, epsilon=0.001, center=True, scale=True)(conv12)

        up_conv12 = layers.UpSampling2D(size=(2, 2))(conv12)
        up_conv2 = layers.Conv2D(336, kernel_size=5, padding='same',
                                 kernel_initializer='he_normal', activation='relu')(up_conv12)
        ch, cw = self.get_crop_shape(conv6, up_conv2)
        crop_conv6 = layers.Cropping2D(cropping=(ch, cw))(conv6)
        up_samp2 = layers.concatenate([crop_conv6, up_conv2], axis=concat_axis)

        conv13 = layers.Conv2D(112, kernel_size=5, padding='same', kernel_initializer='he_normal', activation='relu')(
            up_samp2)
        conv14 = layers.Conv2D(112, kernel_size=5, padding='same', kernel_initializer='he_normal', activation='relu')(
            conv13)
        #conv14 = layers.BatchNormalization(axis=concat_axis, momentum=0.99, epsilon=0.001, center=True, scale=True)(conv14)

        up_conv14 = layers.UpSampling2D(size=(2, 2))(conv14)
        up_conv3 = layers.Conv2D(192, kernel_size=5, padding='same',
                                 kernel_initializer='he_normal', activation='relu')(up_conv14)
        ch, cw = self.get_crop_shape(conv4, up_conv3)
        crop_conv4 = layers.Cropping2D(cropping=(ch, cw))(conv4)
        up_samp3 = layers.concatenate([crop_conv4, up_conv3], axis=concat_axis)

        conv15 = layers.Conv2D(80, kernel_size=5, padding='same', kernel_initializer='he_normal', activation='relu')(
            up_samp3)
        conv16 = layers.Conv2D(80, kernel_size=5, padding='same', kernel_initializer='he_normal', activation='relu')(
            conv15)
        #conv16 = layers.BatchNormalization(axis=concat_axis, momentum=0.99, epsilon=0.001, center=True, scale=True)(conv16)

        up_conv16 = layers.UpSampling2D(size=(2, 2))(conv16)
        up_conv4 = layers.Conv2D(140, kernel_size=5, padding='same',
                                 kernel_initializer='he_normal', activation='relu')(up_conv16)
        ch, cw = self.get_crop_shape(conv2, up_conv4)
        crop_conv2 = layers.Cropping2D(cropping=(ch, cw))(conv2)
        up_samp4 = layers.concatenate([crop_conv2, up_conv4], axis=concat_axis)

        conv21 = layers.Conv2D(40, kernel_size=5, padding='same', kernel_initializer='he_normal', activation='relu')(
            up_samp4)
        conv22 = layers.Conv2D(40, kernel_size=5, padding='same', kernel_initializer='he_normal', activation='relu')(
            conv21)
        #conv22 = layers.BatchNormalization(axis=concat_axis, momentum=0.99, epsilon=0.001, center=True, scale=True)(conv22)

        conv23 = layers.Conv2D(1, kernel_size=1, padding='same', kernel_initializer='he_normal', activation='sigmoid')(
            conv22)
        model = models.Model(inputs=inputs, outputs=conv23)

        model.compile(optimizer=Adam(lr=0.000001), loss=dice_coef_loss,
                      metrics=[dice_coef, obj_recall])
        model.summary()

        return model


