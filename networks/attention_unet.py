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

class AttentionUnet(BaseNetwork):

    def __init__(self,  model_path=None, img_shape=None):

        if model_path is None:
            if img_shape is None:
                raise Exception('If no model path is provided img shape is a mandatory argument.')
            model = self.create_model(img_shape)
        else:
            model = load_model(model_path)

        super().__init__(model, img_shape)


    def create_model(self, img_shape, kernel_size=11):
        concat_axis = 3

        inputs = layers.Input(shape=img_shape)

        stack1 = layers.Conv2D(80, kernel_size=kernel_size, padding='same', kernel_initializer='he_normal', activation='relu')(
            inputs)
        stack2 = layers.Conv2D(80, kernel_size=kernel_size, padding='same', kernel_initializer='he_normal', activation='relu')(
            stack1)
        stack3 = layers.Conv2D(80, kernel_size=kernel_size, padding='same', kernel_initializer='he_normal', activation='relu')(
            stack2)

        stack4 = layers.Conv2D(80, kernel_size=kernel_size, padding='same', kernel_initializer='he_normal',
                               activation='relu')(stack3)
        stack5 = layers.Conv2D(80, kernel_size=kernel_size, padding='same', kernel_initializer='he_normal',
                               activation='relu')(stack4)

        conv1 = layers.Conv2D(80, kernel_size=kernel_size, padding='same', kernel_initializer='he_normal', activation='relu')(
            stack5)
        conv2 = layers.Conv2D(80, kernel_size=kernel_size, padding='same', kernel_initializer='he_normal', activation='relu')(
            conv1)

        maxpool1 = layers.MaxPool2D(pool_size=(2, 2))(conv2)

        conv3 = layers.Conv2D(80, kernel_size=kernel_size, padding='same', kernel_initializer='he_normal', activation='relu')(
            maxpool1)
        conv4 = layers.Conv2D(80, kernel_size=kernel_size, padding='same', kernel_initializer='he_normal', activation='relu')(
            conv3)
        maxpool2 = layers.MaxPool2D(pool_size=(2, 2))(conv4)

        conv5 = layers.Conv2D(100, kernel_size=kernel_size, padding='same', kernel_initializer='he_normal', activation='relu')(
            maxpool2)
        conv6 = layers.Conv2D(100, kernel_size=kernel_size, padding='same', kernel_initializer='he_normal', activation='relu')(
            conv5)
        maxpool3 = layers.MaxPool2D(pool_size=(2, 2))(conv6)


        conv11 = layers.Conv2D(180, kernel_size=kernel_size, padding='same', kernel_initializer='he_normal', activation='relu')(
            maxpool3)
        conv12 = layers.Conv2D(180, kernel_size=kernel_size, padding='same', kernel_initializer='he_normal', activation='relu')(
            conv11)

        gating1 = self.UnetGatingSignal(conv12, is_batchnorm=True)
        attn1 = self.AttnGatingBlock(conv6, gating1, 180)
        up_samp1 = layers.Conv2DTranspose(180, kernel_size, strides=(2, 2), padding='same', activation="relu")(conv12)
        concat1 = layers.concatenate([up_samp1, attn1], axis=3)

        gating2 = self.UnetGatingSignal(concat1, is_batchnorm=True)
        attn2 = self.AttnGatingBlock(conv4, gating2, 100)
        up_samp2 = layers.Conv2DTranspose(100, kernel_size, strides=(2, 2), padding='same', activation="relu")(concat1)
        concat2 = layers.concatenate([up_samp2, attn2], axis=3)

        gating3 = self.UnetGatingSignal(concat2, is_batchnorm=True)
        attn3 = self.AttnGatingBlock(conv2, gating3, 100)
        up_samp3 = layers.Conv2DTranspose(100, kernel_size, strides=(2, 2), padding='same', activation="relu")(concat2)
        concat3 = layers.concatenate([up_samp3, attn3], axis=3)

        #ch, cw = self.get_crop_shape(conv6, up_conv2)
        #crop_conv6 = layers.Cropping2D(cropping=(ch, cw))(conv6)
        #ch, cw = self.get_crop_shape(conv4, up_conv3)
        #crop_conv4 = layers.Cropping2D(cropping=(ch, cw))(conv4)
        #ch, cw = self.get_crop_shape(conv2, up_conv4)
        #crop_conv2 = layers.Cropping2D(cropping=(ch, cw))(conv2)

        conv21 = layers.Conv2D(80, kernel_size=kernel_size, padding='same', kernel_initializer='he_normal', activation='relu')(
            concat3)
        conv22 = layers.Conv2D(80, kernel_size=kernel_size, padding='same', kernel_initializer='he_normal', activation='relu')(
            conv21)

        final_stack_1 = layers.Conv2D(80, kernel_size=kernel_size, padding='same', kernel_initializer='he_normal', activation='relu')(
            conv22)
        final_stack_2 = layers.Conv2D(60, kernel_size=kernel_size, padding='same', kernel_initializer='he_normal',
                                      activation='relu')(final_stack_1)
        final_stack_3 = layers.Conv2D(40, kernel_size=kernel_size, padding='same', kernel_initializer='he_normal',
                                      activation='relu')(final_stack_2)

        conv23 = layers.Conv2D(1, kernel_size=1, padding='same', kernel_initializer='he_normal', activation='sigmoid')(
            final_stack_3)
        model = models.Model(inputs=inputs, outputs=conv23)

        model.compile(optimizer=Adam(lr=0.00001), loss=dice_coef_loss,
                      metrics=[dice_coef, obj_recall])
        model.summary()

        return model



    def AttnGatingBlock(self,x, g, inter_shape):
            shape_x = K.int_shape(x)  # 32
            shape_g = K.int_shape(g)  # 16

            theta_x = layers.Conv2D(inter_shape, (5, 5), strides=(2, 2), padding='same')(x)  # 16
            shape_theta_x = K.int_shape(theta_x)

            phi_g = layers.Conv2D(inter_shape, (1, 1), padding='same')(g)
            upsample_g = layers.Conv2DTranspose(inter_shape, (5, 5),strides=(shape_theta_x[1] // shape_g[1], shape_theta_x[2] // shape_g[2]),padding='same')(phi_g)  # 16

            concat_xg = layers.add([upsample_g, theta_x])
            act_xg = layers.Activation('relu')(concat_xg)
            psi = layers.Conv2D(1, (1, 1), padding='same')(act_xg)
            sigmoid_xg = layers.Activation('sigmoid')(psi)
            shape_sigmoid = K.int_shape(sigmoid_xg)
            upsample_psi = layers.UpSampling2D(size=(shape_x[1] // shape_sigmoid[1], shape_x[2] // shape_sigmoid[2]))(sigmoid_xg)  # 32

            upsample_psi = self.expend_as(upsample_psi, shape_x[3])

            y = layers.multiply([upsample_psi, x])

            result = layers.Conv2D(shape_x[3], (1, 1), padding='same')(y)
            result_bn = layers.BatchNormalization()(result)
            return result_bn

    def UnetGatingSignal(self, input, is_batchnorm=False):
        shape = K.int_shape(input)
        x = layers.Conv2D(shape[3] * 2, (1, 1), strides=(1, 1), padding="same")(input)
        if is_batchnorm:
            x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        return x

    def expend_as(self, tensor, rep):
        my_repeat = layers.Lambda(lambda x, repnum: K.repeat_elements(x, repnum, axis=3), arguments={'repnum': rep})(tensor)
        return my_repeat

