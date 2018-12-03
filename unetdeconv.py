from keras import models
from keras import layers
from contextlib import redirect_stdout
import os
from keras.callbacks import ModelCheckpoint, TensorBoard
from sklearn.model_selection import train_test_split
from keras.models import load_model
from keras.optimizers import Adam, SGD
from metrics import dice_coef, dice_coef_loss, weighted_crossentropy, predicted_count, ground_truth_count, ground_truth_sum, predicted_sum
from keras.losses import binary_crossentropy
import cv2
import numpy as np
import custom_layers
import keras.backend as K
import math
import pickle
import gc
import psutil
import tensorflow as tf
from basenetwork import BaseNetwork

class UnetDeconv(BaseNetwork):

    def __init__(self, model_paths=None, img_shape=None):

        if model_paths is None:
            if img_shape is None:
                raise ValueError('If no model path is provided img shape is a mandatory argument.')
            model, deconv_model = self.create_model(img_shape)
        elif (isinstance(model_paths, tuple) or isinstance(model_paths, list)) and len(model_paths) == 2:
            model = load_model(model_paths[0])
            deconv_model = load_model(model_paths[1])
        else:
            raise ValueError("model_paths must be an array_like with len = 2, containing both paths for model"
                             "and deconv model.")

        super().__init__(model)
        self.deconv_model = deconv_model


    def create_model(self, img_shape):

        concat_axis = 3

        inputs = layers.Input(shape=img_shape)

        conv1 = layers.Conv2D(64, kernel_size=5, padding='same', kernel_initializer='he_normal', activation='relu')(inputs)
        conv2 = layers.Conv2D(64, kernel_size=5, padding='same', kernel_initializer='he_normal', activation='relu')(conv1)
        maxpool1, switches_mask1 = custom_layers.MaxPoolingWithArgmax2D(pool_size=(2, 2))(conv2)

        conv3 = layers.Conv2D(96, kernel_size=3, padding='same', kernel_initializer='he_normal', activation='relu')(maxpool1)
        conv4 = layers.Conv2D(96, kernel_size=3, padding='same', kernel_initializer='he_normal', activation='relu')(conv3)
        maxpool2, switches_mask2 = custom_layers.MaxPoolingWithArgmax2D(pool_size=(2, 2))(conv4)

        conv5 = layers.Conv2D(128, kernel_size=3, padding='same', kernel_initializer='he_normal', activation='relu')(maxpool2)
        conv6 = layers.Conv2D(128, kernel_size=3, padding='same', kernel_initializer='he_normal', activation='relu')(conv5)
        maxpool3, switches_mask3 = custom_layers.MaxPoolingWithArgmax2D(pool_size=(2, 2))(conv6)

        conv7 = layers.Conv2D(256, kernel_size=3, padding='same', kernel_initializer='he_normal', activation='relu')(maxpool3)
        conv8 = layers.Conv2D(256, kernel_size=4, padding='same', kernel_initializer='he_normal', activation='relu')(conv7)
        maxpool4, switches_mask4 = custom_layers.MaxPoolingWithArgmax2D(pool_size=(2, 2))(conv8)

        conv9 = layers.Conv2D(512, kernel_size=3, padding='same', kernel_initializer='he_normal', activation='relu')(maxpool4)
        conv10 = layers.Conv2D(512, kernel_size=3, padding='same', kernel_initializer='he_normal', activation='relu')(conv9)

        up_conv10 = layers.UpSampling2D(size=(2, 2))(conv10)
        up_samp1 = layers.concatenate([conv8, up_conv10], axis=concat_axis)

        conv12 = layers.Conv2D(256, kernel_size=3, padding='same', kernel_initializer='he_normal', activation='relu')(up_samp1)
        conv13 = layers.Conv2D(256, kernel_size=3, padding='same', kernel_initializer='he_normal', activation='relu')(conv12)

        up_conv13 = layers.UpSampling2D(size=(2, 2))(conv13)
        up_samp2 = layers.concatenate([conv6, up_conv13], axis=concat_axis)

        conv15 = layers.Conv2D(128, kernel_size=3, padding='same', kernel_initializer='he_normal', activation='relu')(up_samp2)
        conv16 = layers.Conv2D(128, kernel_size=3, padding='same', kernel_initializer='he_normal', activation='relu')(conv15)

        up_conv16 = layers.UpSampling2D(size=(2, 2))(conv16)
        up_samp3 = layers.concatenate([conv4, up_conv16], axis=concat_axis)

        conv18 = layers.Conv2D(96, kernel_size=3, padding='same', kernel_initializer='he_normal', activation='relu')(up_samp3)
        conv19 = layers.Conv2D(96, kernel_size=3, padding='same', kernel_initializer='he_normal', activation='relu')(conv18)

        up_conv19 = layers.UpSampling2D(size=(2, 2))(conv19)
        up_samp4 = layers.concatenate([conv2, up_conv19], axis=concat_axis)

        conv21 = layers.Conv2D(64, kernel_size=3, padding='same', kernel_initializer='he_normal', activation='relu')(up_samp4)
        conv22 = layers.Conv2D(64, kernel_size=3, padding='same', kernel_initializer='he_normal', activation='relu')(conv21)

        conv23 = layers.Conv2D(1, kernel_size=1, padding='same', kernel_initializer='he_normal', activation='sigmoid')(conv22)

        deconv_10 = layers.Conv2DTranspose(512, kernel_size=3, padding='same', kernel_initializer='he_normal', activation='relu')(conv10)
        deconv_09 = layers.Conv2DTranspose(512, kernel_size=3, padding='same', kernel_initializer='he_normal', activation='relu')(deconv_10)

        unpool_04 = layers.Lambda(custom_layers.unpooling_with_argmax2D,
                            arguments={"poolsize": (2, 2), "argmax": switches_mask4},
                            output_shape=K.int_shape(switches_mask4)[1:])(deconv_09)

        deconv_08 = layers.Conv2DTranspose(256, kernel_size=4, padding='same', kernel_initializer='he_normal',
                                           activation='relu')(unpool_04)
        deconv_07 = layers.Conv2DTranspose(256, kernel_size=3, padding='same', kernel_initializer='he_normal',
                                           activation='relu')(deconv_08)

        unpool_03 = layers.Lambda(custom_layers.unpooling_with_argmax2D,
                                  arguments={"poolsize": (2, 2), "argmax": switches_mask3},
                                  output_shape=K.int_shape(switches_mask3)[1:])(deconv_07)

        deconv_06 = layers.Conv2DTranspose(128, kernel_size=3, padding='same', kernel_initializer='he_normal',
                                           activation='relu')(unpool_03)
        deconv_05 = layers.Conv2DTranspose(128, kernel_size=3, padding='same', kernel_initializer='he_normal',
                                           activation='relu')(deconv_06)

        unpool_02 = layers.Lambda(custom_layers.unpooling_with_argmax2D,
                                  arguments={"poolsize": (2, 2), "argmax": switches_mask2},
                                  output_shape=K.int_shape(switches_mask2)[1:])(deconv_05)

        deconv_04 = layers.Conv2DTranspose(96, kernel_size=3, padding='same', kernel_initializer='he_normal',
                                           activation='relu')(unpool_02)
        deconv_03 = layers.Conv2DTranspose(96, kernel_size=3, padding='same', kernel_initializer='he_normal',
                                           activation='relu')(deconv_04)

        unpool_01 = layers.Lambda(custom_layers.unpooling_with_argmax2D,
                                  arguments={"poolsize": (2, 2), "argmax": switches_mask1},
                                  output_shape=K.int_shape(switches_mask1)[1:])(deconv_03)

        deconv_02 = layers.Conv2DTranspose(64, kernel_size=5, padding='same', kernel_initializer='he_normal',
                                           activation='relu')(unpool_01)
        deconv_01 = layers.Conv2DTranspose(64, kernel_size=5, padding='same', kernel_initializer='he_normal',
                                           activation='relu')(deconv_02)

        model = models.Model(inputs=inputs, outputs=conv23)

        deconv_model = models.Model(inputs=inputs, outputs=deconv_01)

        model.compile(optimizer=Adam(lr=0.000001), loss=dice_coef_loss, metrics=[dice_coef, binary_crossentropy, weighted_crossentropy,
                                                                                   predicted_count, predicted_sum, ground_truth_count,
                                                                                 ground_truth_sum])
        model.summary()

        return model, deconv_model


    def train(self, X, y, test_size, training_name, base_path, epochs=10, batch_size=32):

        self.create_folders(training_name, base_path, viz_path_flag=True)

        checkpointer = ModelCheckpoint(filepath=self.full_paths_dict["weights_path"],
                                       save_best_only=True,
                                       verbose=1)

        tensorboard_callback = TensorBoard(log_dir=self.full_paths_dict["log_path"],
                                           batch_size=batch_size,
                                           write_graph=False,
                                           write_grads=False,
                                           write_images=False,
                                           embeddings_freq=0,
                                           embeddings_layer_names=None,
                                           embeddings_metadata=None)

        X_train, X_test, y_train, y_test = train_test_split(X,
                                                            y,
                                                            test_size=test_size)

        del X, y
        gc.collect()

        fit_specs = {
            'epochs': epochs,
            'batch_size': batch_size,
            'test_size': test_size

        }
        self.save_specs(self.full_paths_dict['specs_path'], fit_specs)


        self.model.fit(X_train, y_train,
                       batch_size=batch_size,
                       callbacks=[checkpointer, tensorboard_callback],
                       epochs=epochs,
                       validation_data=(X_test, y_test),
                       verbose=1)


    def train_with_generator(self, X, y, test_size, training_name, base_path, epochs=10, batch_size=32):

        self.create_folders(training_name, base_path, viz_path_flag=True)

        checkpointer = ModelCheckpoint(filepath=self.full_paths_dict["weights_path"],
                                       save_best_only=True,
                                       verbose=1)

        tensorboard_callback = TensorBoard(log_dir=self.full_paths_dict["log_path"],
                                           batch_size=batch_size,
                                           write_graph=False,
                                           write_grads=False,
                                           write_images=False,
                                           embeddings_freq=0,
                                           embeddings_layer_names=None,
                                           embeddings_metadata=None)

        fit_specs = {
            'epochs': epochs,
            'batch_size': batch_size,
            'test_size': test_size

        }
        self.save_specs(self.full_paths_dict['specs_path'], fit_specs)

        X_train, X_test, y_train, y_test = train_test_split(X,
                                                            y,
                                                            test_size=test_size)

        del X, y
        gc.collect()
        print(psutil.Process().memory_info().rss / 2 ** 30)


        number_batches = self.save_batch_files(X_train, y_train, base_path, batch_size, "train")
        _ = self.save_batch_files(X_test, y_test, base_path, batch_size, "test")

        del X_train, y_train, X_test, y_test
        gc.collect()
        print("Pre fit_generator: ", psutil.Process().memory_info().rss / 2 ** 30)


        self.model.fit_generator(self.train_generator(base_path, number_batches, "train"),
                       steps_per_epoch=1,
                       callbacks=[checkpointer, tensorboard_callback],
                       epochs=epochs,
                       validation_steps=1,
                       validation_data=self.train_generator(base_path, number_batches, "test"),
                       verbose=1)


    def visualize_activations(self, data, labels, batch_size=1):

        output_path = self.full_paths_dict['output_path']

        predictions = self.deconv_model.predict(data, batch_size=batch_size, verbose=1)

        print("Predictions deconv shape", predictions.shape)
        for index, (pred, original, label) in enumerate(zip(predictions, data, labels)):
            cv2.imwrite(output_path + str(index) + '_original_deconv_activations_layer_4' + '.png', original * 255)
            cv2.imwrite(output_path + str(index) + '_label_deconv_activations_layer_4' + '.png', label * 255)
            for channel in range(predictions.shape[-1]):
                file_name = output_path + str(index) + '_label_deconv_activations_layer_4_' + "chan_" + str(channel) + '.png'
                cv2.imwrite(file_name, pred[:, :, channel] * 255)

