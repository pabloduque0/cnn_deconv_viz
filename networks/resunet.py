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

class ResUnet(BaseNetwork):

    def __init__(self,  model_path=None, img_shape=None):

        if model_path is None:
            if img_shape is None:
                raise Exception('If no model path is provided img shape is a mandatory argument.')
            model = self.create_model(img_shape)
        else:
            model = load_model(model_path)

        super().__init__(model, img_shape)


    def create_model(self, img_shape, kernel_size=5):
        inputs = layers.Input(shape=img_shape)

        stack1 = layers.Conv2D(40, kernel_size=kernel_size, padding='same', kernel_initializer='he_normal', activation='relu')(
            inputs)
        stack2 = layers.Conv2D(40, kernel_size=kernel_size, padding='same', kernel_initializer='he_normal', activation='relu')(
            stack1)
        stack3 = layers.Conv2D(40, kernel_size=kernel_size, padding='same', kernel_initializer='he_normal', activation='relu')(
            stack2)

        stack4 = layers.Conv2D(40, kernel_size=kernel_size, padding='same', kernel_initializer='he_normal',
                               activation='relu')(stack3)
        stack5 = layers.Conv2D(40, kernel_size=kernel_size, padding='same', kernel_initializer='he_normal',
                               activation='relu')(stack4)

        res_out_1 = res_block(stack5, filters=80, kernel_size=kernel_size, dilations=(1, 3, 15, 31))
        strided_1 = layers.Conv2D(80, kernel_size=1, strides=2, padding='same', kernel_initializer='he_normal', activation='relu')(res_out_1)
        res_out_2 = res_block(strided_1, filters=128, kernel_size=kernel_size, dilations=(1, 3, 15, 31))
        strided_2 = layers.Conv2D(128, kernel_size=1, strides=2, padding='same', kernel_initializer='he_normal',
                                  activation='relu')(res_out_2)
        res_out_3 = res_block(strided_2, filters=256, kernel_size=kernel_size, dilations=(1, 3, 15))
        strided_3 = layers.Conv2D(256, kernel_size=1, strides=2, padding='same', kernel_initializer='he_normal',
                                  activation='relu')(res_out_3)
        res_out_4 = res_block(strided_3, filters=256, kernel_size=kernel_size, dilations=(1, 3, 15))
        upsample_1 = layers.UpSampling2D(size=(2, 2))(res_out_4)
        comb_1 = combine_block(upsample_1, res_out_3, 128)
        res_out_5 = res_block(comb_1, filters=128, kernel_size=kernel_size, dilations=(1, 3, 15))

        upsample_2 = layers.UpSampling2D(size=(2, 2))(res_out_5)
        comb_2 = combine_block(upsample_2, res_out_2, 80)
        res_out_6 = res_block(comb_2, filters=80, kernel_size=kernel_size, dilations=(1, 3, 15, 31))

        upsample_3 = layers.UpSampling2D(size=(2, 2))(res_out_6)
        comb_3 = combine_block(upsample_3, res_out_1, 40)
        res_out_5 = res_block(comb_3, filters=40, kernel_size=kernel_size, dilations=(1, 3, 15, 31))

        conv23 = layers.Conv2D(1, kernel_size=1, padding='same',
                               kernel_initializer='he_normal', activation='sigmoid')(res_out_5)
        model = models.Model(inputs=inputs, outputs=conv23)

        model.compile(optimizer=Adam(lr=0.000001), loss=dice_coef_loss,
                      metrics=[dice_coef, obj_recall])
        model.summary()

        return model


def combine_block(input_tensor, concat_tensor, filters):

    output = layers.Activation("relu")(input_tensor)
    output = layers.concatenate([output, concat_tensor], axis=-1)
    output = layers.Conv2D(filters, kernel_size=1)(output)
    return output

def res_block(input_tensor, filters, kernel_size, dilations):

    dilation_outputs = []
    for dilation in dilations:
        dilation_out = layers.BatchNormalization()(input_tensor)
        dilation_out = layers.Activation("relu")(dilation_out)
        dilation_out = layers.Conv2D(filters, kernel_size=kernel_size, dilation_rate=dilation,
                                     padding='same', kernel_initializer='he_normal')(dilation_out)
        dilation_out = layers.BatchNormalization()(dilation_out)
        dilation_out = layers.Activation("relu")(dilation_out)
        dilation_out = layers.Conv2D(filters, kernel_size=kernel_size, dilation_rate=dilation,
                                     padding='same', kernel_initializer='he_normal')(dilation_out)
        dilation_outputs.append(dilation_out)

    output = layers.Add()(dilation_outputs)

    return output

