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

class UnetDeconv():

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

        self.model = model
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

        deconv_10 = layers.Conv2DTranspose(1, kernel_size=1, padding='same', kernel_initializer='he_normal', activation='relu')(conv10)
        deconv_09 = layers.Conv2DTranspose(64, kernel_size=3, padding='same', kernel_initializer='he_normal', activation='relu')(deconv_10)

        unpool_04 = layers.Lambda(custom_layers.unpooling_with_argmax2D,
                            arguments={"poolsize": (2, 2), "argmax": switches_mask4},
                            output_shape=custom_layers.unpoolingMask2D_output_shape)(deconv_09)

        deconv_08 = layers.Conv2DTranspose(1, kernel_size=1, padding='same', kernel_initializer='he_normal',
                                           activation='relu')(unpool_04)
        deconv_07 = layers.Conv2DTranspose(64, kernel_size=3, padding='same', kernel_initializer='he_normal',
                                           activation='relu')(deconv_08)

        unpool_03 = layers.Lambda(custom_layers.unpooling_with_argmax2D,
                                  arguments={"poolsize": (2, 2), "argmax": switches_mask3},
                                  output_shape=custom_layers.unpoolingMask2D_output_shape)(deconv_07)

        deconv_06 = layers.Conv2DTranspose(1, kernel_size=1, padding='same', kernel_initializer='he_normal',
                                           activation='relu')(unpool_03)
        deconv_05 = layers.Conv2DTranspose(64, kernel_size=3, padding='same', kernel_initializer='he_normal',
                                           activation='relu')(deconv_06)

        unpool_02 = layers.Lambda(custom_layers.unpooling_with_argmax2D,
                                  arguments={"poolsize": (2, 2), "argmax": switches_mask2},
                                  output_shape=custom_layers.unpoolingMask2D_output_shape)(deconv_05)

        deconv_04 = layers.Conv2DTranspose(1, kernel_size=1, padding='same', kernel_initializer='he_normal',
                                           activation='relu')(unpool_02)
        deconv_03 = layers.Conv2DTranspose(64, kernel_size=3, padding='same', kernel_initializer='he_normal',
                                           activation='relu')(deconv_04)

        unpool_01 = layers.Lambda(custom_layers.unpooling_with_argmax2D,
                                  arguments={"poolsize": (2, 2), "argmax": switches_mask1},
                                  output_shape=custom_layers.unpoolingMask2D_output_shape)(deconv_03)

        deconv_02 = layers.Conv2DTranspose(1, kernel_size=1, padding='same', kernel_initializer='he_normal',
                                           activation='relu')(unpool_01)
        deconv_01 = layers.Conv2DTranspose(64, kernel_size=3, padding='same', kernel_initializer='he_normal',
                                           activation='relu')(deconv_02)


        model = models.Model(inputs=inputs, outputs=conv23)

        deconv_model = models.Model(inputs=inputs, outputs=deconv_01)

        model.compile(optimizer=Adam(lr=0.000001), loss=dice_coef_loss, metrics=[dice_coef, binary_crossentropy, weighted_crossentropy,
                                                                                   predicted_count, predicted_sum, ground_truth_count,
                                                                                 ground_truth_sum])
        model.summary()

        return model, deconv_model


    def save_specs(self, specs_path, fit_specs):

        with open(specs_path, 'w') as file:
            with redirect_stdout(file):
                self.model.summary()

        fit_specs_file = specs_path[:-4] + 'fit_specs.txt'

        with open(fit_specs_file, 'w') as fit_file:
            for key, value in fit_specs.items():
                fit_file.write(key + ': ' + str(value) + '\n')

    def create_folders(self, training_name, base_path):

        model_path = base_path + "/models/" + training_name
        if not os.path.exists(model_path):
            os.makedirs(model_path)

        v = 0
        weights_path = model_path + "/model_0.hdf5"
        if os.path.exists(weights_path):
            try:
                v = int(weights_path.split("_")[-1].replace(".hdf5", "")) + 1
            except ValueError:
                v = 1
            weights_path = model_path + "/model_{}.hdf5".format(v)

        log_path = base_path + "/logs/" + training_name + '/'
        if not os.path.exists(log_path):
            os.makedirs(log_path)

        specs_path = log_path + "/specs_{}.txt".format(v)

        return {"log_path": log_path, "weights_path": weights_path,
                "specs_path": specs_path}

    def train(self, X, y, test_size, training_name, base_path, epochs=10, batch_size=32):

        paths = self.create_folders(training_name, base_path)

        checkpointer = ModelCheckpoint(filepath=paths["weights_path"],
                                       save_best_only=True,
                                       verbose=1)

        tensorboard_callback = TensorBoard(log_dir=paths["log_path"],
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
        self.save_specs(paths['specs_path'], fit_specs)


        self.model.fit(X_train, y_train,
                       batch_size=batch_size,
                       callbacks=[checkpointer, tensorboard_callback],
                       epochs=epochs,
                       validation_data=(X_test, y_test),
                       verbose=1)


    def train_with_generator(self, X, y, test_size, training_name, base_path, epochs=10, batch_size=32):

        paths = self.create_folders(training_name, base_path)

        checkpointer = ModelCheckpoint(filepath=paths["weights_path"],
                                       save_best_only=True,
                                       verbose=1)

        tensorboard_callback = TensorBoard(log_dir=paths["log_path"],
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
        self.save_specs(paths['specs_path'], fit_specs)

        X_train, X_test, y_train, y_test = train_test_split(X,
                                                            y,
                                                            test_size=test_size)

        del X, y
        gc.collect()
        print(psutil.Process().memory_info().rss / 2 ** 30)


        number_batches = self.save_batch_files(X_train, y_train, base_path, batch_size, "train")
        _ = self.save_batch_files(X_test, y_test, base_path, batch_size, "test")


        self.model.fit_generator(self.train_generator(base_path, number_batches, "train"),
                       callbacks=[checkpointer, tensorboard_callback],
                       epochs=epochs,
                       validation_data=self.train_generator(base_path, number_batches, "test"),
                       verbose=1)


    def save_batch_files(self, data, labels, base_path, batch_size, _type):

        data_output_path = base_path + "/data_batches_" + _type + "/"
        if not os.path.exists(data_output_path):
            os.makedirs(data_output_path)

        _range = math.ceil(data.shape[0]/batch_size)
        counter = 0
        for id in range(_range):
            tuple_to_save = (data[:, :, :, counter:counter+batch_size], labels[:, :, :, counter:counter+batch_size])
            file_name = data_output_path + "data_and_labels_" + str(id) + ".pk"
            pickle.dump(tuple_to_save, open(file_name, "wb"))
            del tuple_to_save
            gc.collect()

        return _range


    def train_generator(self, base_path, number_batches, _type):

        for id in range(number_batches):
            file_name = base_path + "/data_batches_" + _type + "/" + "data_and_labels_" + str(id) + ".pk"
            loaded_tuple = pickle.load(open(file_name, "rb"))
            yield loaded_tuple


    def predict_and_save(self, data, labels, output_path, batch_size=1):

        if not output_path.endswith('/'):
            output_path += '/'

        if not os.path.exists(output_path):
            os.makedirs(output_path)

        predictions = self.model.predict(data, batch_size=batch_size, verbose=1)

        for index, (pred, original, label) in enumerate(zip(predictions, data, labels)):
            print(len(np.flatnonzero(pred)))
            cv2.imwrite(output_path + 'original_' + str(index) + '.png', original * 255)
            cv2.imwrite(output_path + 'prediction_' + str(index) + '.png', pred * 255)
            cv2.imwrite(output_path + 'label_' + str(index) + '.png', label * 255)


    def visualize_activations(self, data, labels, output_path, batch_size=1):

        predictions = self.deconv_model.predict(data, batch_size=batch_size, verbose=1)

        print("Predictions deconv shape", predictions.shape)
        for index, (pred, original, label) in enumerate(zip(predictions, data, labels)):
            cv2.imwrite(output_path + 'deconv_activations_layer_4' + '.png', label * 255)
