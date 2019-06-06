from contextlib import redirect_stdout
import os
from keras.callbacks import ModelCheckpoint, TensorBoard
from sklearn.model_selection import train_test_split
import cv2
import numpy as np
import math
import pickle
import gc

class BaseNetwork():

    def __init__(self, model):
        self.model = model
        self.full_paths_dict = None
        self.training_name = None

    def get_crop_shape(self, target, refer):
        # width, the 3rd dimension
        cw = (target.get_shape()[2] - refer.get_shape()[2]).value
        assert (cw >= 0)
        if cw % 2 != 0:
            cw1, cw2 = int(cw / 2), int(cw / 2) + 1
        else:
            cw1, cw2 = int(cw / 2), int(cw / 2)
        # height, the 2nd dimension
        ch = (target.get_shape()[1] - refer.get_shape()[1]).value
        assert (ch >= 0)
        if ch % 2 != 0:
            ch1, ch2 = int(ch / 2), int(ch / 2) + 1
        else:
            ch1, ch2 = int(ch / 2), int(ch / 2)

        return (ch1, ch2), (cw1, cw2)

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


    def train(self, X, y, validation_data, training_name, base_path, epochs=10, batch_size=32):

        self.create_folders(training_name, base_path)
        self.training_name = training_name

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
            'batch_size': batch_size

        }
        self.save_specs(self.full_paths_dict['specs_path'], fit_specs)


        self.model.fit(X, y,
                       batch_size=batch_size,
                       callbacks=[checkpointer, tensorboard_callback],
                       epochs=epochs,
                       validation_data=validation_data,
                       verbose=1)


    def train_with_generator(self, X, y, validation_data, training_name, base_path, epochs=10, batch_size=32):

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
            'batch_size': batch_size

        }
        self.save_specs(self.full_paths_dict['specs_path'], fit_specs)

        gc.collect()
        number_batches = self.save_batch_files(X, y, base_path, batch_size, "train")
        _ = self.save_batch_files(validation_data[0], validation_data[1], base_path, batch_size, "test")

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

        print("Saving output image from validation...")
        for index, (pred, original, label) in enumerate(zip(predictions, data, labels)):
            original = (original - np.min(original)) * 255 / (np.max(original) - np.min(original))

            cv2.imwrite(output_path + 'original_' + str(index) + '.png',
                        np.concatenate([original[:, :, 0], original[:, :, 1]], axis=1))
            cv2.imwrite(output_path + 'prediction_' + str(index) + '.png', pred * 255)
            cv2.imwrite(output_path + 'label_' + str(index) + '.png', label * 255)
