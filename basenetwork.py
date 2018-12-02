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
import progressbar


class BaseNetwork():

    def __init__(self, model):
        self.model = model
        self.full_paths_dict = None


    def save_specs(self, specs_path, fit_specs):

        with open(specs_path, 'w') as file:
            with redirect_stdout(file):
                self.model.summary()

        fit_specs_file = specs_path[:-4] + 'fit_specs.txt'

        with open(fit_specs_file, 'w') as fit_file:
            for key, value in fit_specs.items():
                fit_file.write(key + ': ' + str(value) + '\n')

    def create_folders(self, training_name, base_path, output_path_flag=True, viz_path_flag=False):

        full_paths_dict = {}

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

        full_paths_dict["weights_path"] = weights_path

        log_path = base_path + "/logs/" + training_name + '/'
        if not os.path.exists(log_path):
            os.makedirs(log_path)
        full_paths_dict["log_path"] = log_path

        specs_path = log_path + "/specs_{}.txt".format(v)
        full_paths_dict["specs_path"] = specs_path

        if output_path_flag:
            output_path = base_path + "/output/" + training_name + '/'
            if not os.path.exists(output_path):
                os.makedirs(output_path)
            full_paths_dict["output_path"] = output_path

        if viz_path_flag:
            viz_path = base_path + "/viz_output/" + training_name + '/'
            if not os.path.exists(viz_path):
                os.makedirs(viz_path)
            full_paths_dict["viz_path"] = viz_path

        self.full_paths_dict = full_paths_dict


    def train(self, X, y, test_size, training_name, base_path, epochs=10, batch_size=32):

        self.create_folders(training_name, base_path)

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

        self.create_folders(training_name, base_path)

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


    def save_batch_files(self, data, labels, base_path, batch_size, _type):

        data_output_path = base_path + "/data_batches_" + _type + "/"
        if not os.path.exists(data_output_path):
            os.makedirs(data_output_path)

        _range = math.ceil(data.shape[0]/batch_size)
        counter = 0
        bar = progressbar.ProgressBar()
        for id in bar(range(_range)):
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


    def predict_and_save(self, data, labels, batch_size=1):

        output_path = self.full_paths_dict['output_path']

        predictions = self.model.predict(data, batch_size=batch_size, verbose=1)

        for index, (pred, original, label) in enumerate(zip(predictions, data, labels)):
            print(len(np.flatnonzero(pred)))
            cv2.imwrite(output_path + 'original_' + str(index) + '.png', original * 255)
            cv2.imwrite(output_path + 'prediction_' + str(index) + '.png', pred * 255)
            cv2.imwrite(output_path + 'label_' + str(index) + '.png', label * 255)
