from keras import models
from keras import layers
import os
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.models import load_model
from keras.optimizers import Adam
from networks.metrics import dice_coef, dice_coef_loss, weighted_crossentropy, predicted_count, \
    ground_truth_count, ground_truth_sum, predicted_sum, recall
from keras.losses import binary_crossentropy
import cv2
import numpy as np
from networks import custom_layers
import keras.backend as K
from networks.basenetwork import BaseNetwork

class UnetDeconv(BaseNetwork):

    def __init__(self, model_paths=None, img_shape=None, deconv_shape=None):

        if model_paths is None:
            if img_shape is None or deconv_shape is None:
                raise ValueError('If no model path is provided img shape is a mandatory argument.')
            model, deconv_model = self.create_model(img_shape, deconv_shape)
        elif (isinstance(model_paths, tuple) or isinstance(model_paths, list)) and len(model_paths) == 2:
            model = load_model(model_paths[0])
            deconv_model = load_model(model_paths[1])
        else:
            raise ValueError("model_paths must be an array_like with len = 2, containing both paths for model"
                             "and deconv model.")

        super().__init__(model)
        self.deconv_models = deconv_model


    def create_model(self, img_shape, deconv_shape):

        concat_axis = 3

        inputs = layers.Input(shape=img_shape)

        deconv_inputs = layers.Input(shape=deconv_shape)
        print("INSIDE", deconv_inputs)
        conv1_layer = layers.Conv2D(64, kernel_size=5, padding='same',
                              kernel_initializer='he_normal', activation='relu')
        conv1 = conv1_layer(inputs)
        conv2_layer = layers.Conv2D(64, kernel_size=5, padding='same',
                              kernel_initializer='he_normal', activation='relu')
        conv2 = conv2_layer(conv1)
        maxpool1_layer = custom_layers.MaxPoolingWithArgmax2D(pool_size=(2, 2))
        maxpool1, switches_mask1 = maxpool1_layer(conv2)

        conv3_layer = layers.Conv2D(96, kernel_size=5, padding='same',
                      kernel_initializer='he_normal', activation='relu')
        conv3 = conv3_layer(maxpool1)
        conv4_layer = layers.Conv2D(96, kernel_size=5, padding='same',
                              kernel_initializer='he_normal', activation='relu')
        conv4 = conv4_layer(conv3)
        maxpool2_layer = custom_layers.MaxPoolingWithArgmax2D(pool_size=(2, 2))
        maxpool2, switches_mask2 = maxpool2_layer(conv4)

        conv5_layer = layers.Conv2D(128, kernel_size=5, padding='same',
                      kernel_initializer='he_normal', activation='relu')
        conv5 = conv5_layer(maxpool2)
        conv6_layer = layers.Conv2D(128, kernel_size=5, padding='same',
                              kernel_initializer='he_normal', activation='relu')
        conv6 = conv6_layer(conv5)
        maxpool3_layer = custom_layers.MaxPoolingWithArgmax2D(pool_size=(2, 2))
        maxpool3, switches_mask3 = maxpool3_layer(conv6)

        conv7_layer = layers.Conv2D(256, kernel_size=5, padding='same',
                              kernel_initializer='he_normal', activation='relu')
        conv7 = conv7_layer(maxpool3)
        conv8_layer = layers.Conv2D(256, kernel_size=5, padding='same',
                              kernel_initializer='he_normal', activation='relu')
        conv8 = conv8_layer(conv7)
        maxpool4_layer = custom_layers.MaxPoolingWithArgmax2D(pool_size=(2, 2))
        maxpool4, switches_mask4 = maxpool4_layer(conv8)

        conv9_layer = layers.Conv2D(512, kernel_size=5, padding='same',
                              kernel_initializer='he_normal', activation='relu')
        conv9 = conv9_layer(maxpool4)
        conv10_layer = layers.Conv2D(512, kernel_size=5, padding='same',
                               kernel_initializer='he_normal', activation='relu')
        conv10 = conv10_layer(conv9)

        up_samp1_layer = layers.UpSampling2D(size=(2, 2))
        up_samp1 = up_samp1_layer(conv10)
        conv11_layer = layers.Conv2D(256, kernel_size=5, padding='same',
                               kernel_initializer='he_normal', activation='relu')
        conv11 = conv11_layer(up_samp1)
        concat1 = layers.concatenate([conv8, conv11], axis=concat_axis)

        conv12_layer = layers.Conv2D(256, kernel_size=5, padding='same',
                               kernel_initializer='he_normal', activation='relu')
        conv12 = conv12_layer(concat1)
        conv13_layer = layers.Conv2D(256, kernel_size=5, padding='same',
                               kernel_initializer='he_normal', activation='relu')
        conv13 = conv13_layer(conv12)
        up_samp2_layer = layers.UpSampling2D(size=(2, 2))
        up_samp2 = up_samp2_layer(conv13)
        conv14_layer = layers.Conv2D(128, kernel_size=5, padding='same',
                               kernel_initializer='he_normal', activation='relu')
        conv14 = conv14_layer(up_samp2)
        concat2 = layers.concatenate([conv6, conv14], axis=concat_axis)

        conv15_layer = layers.Conv2D(128, kernel_size=5, padding='same',
                               kernel_initializer='he_normal', activation='relu')
        conv15 = conv15_layer(concat2)
        conv16_layer = layers.Conv2D(128, kernel_size=5, padding='same',
                               kernel_initializer='he_normal', activation='relu')
        conv16 = conv16_layer(conv15)

        up_samp3_layer = layers.UpSampling2D(size=(2, 2))
        up_samp3 = up_samp3_layer(conv16)
        conv17_layer = layers.Conv2D(96, kernel_size=5, padding='same',
                               kernel_initializer='he_normal', activation='relu')
        conv17 = conv17_layer(up_samp3)
        concat3 = layers.concatenate([conv4, conv17], axis=concat_axis)

        conv18_layer = layers.Conv2D(96, kernel_size=5, padding='same',
                               kernel_initializer='he_normal', activation='relu')
        conv18 = conv18_layer(concat3)
        conv19_layer = layers.Conv2D(96, kernel_size=5, padding='same',
                               kernel_initializer='he_normal', activation='relu')
        conv19 = conv19_layer(conv18)

        up_samp4_layer = layers.UpSampling2D(size=(2, 2))
        up_samp4 = up_samp4_layer(conv19)
        conv20_layer = layers.Conv2D(64, kernel_size=5, padding='same',
                               kernel_initializer='he_normal', activation='relu')
        conv20 = conv20_layer(up_samp4)
        concat4 = layers.concatenate([conv2, conv20], axis=concat_axis)

        conv21_layer = layers.Conv2D(64, kernel_size=5, padding='same',
                               kernel_initializer='he_normal', activation='relu')
        conv21 = conv21_layer(concat4)
        conv22_layer = layers.Conv2D(64, kernel_size=5, padding='same',
                               kernel_initializer='he_normal', activation='relu')
        conv22 = conv22_layer(conv21)

        conv23_layer = layers.Conv2D(1, kernel_size=1, padding='same',
                               kernel_initializer='he_normal', activation='sigmoid')
        conv23 = conv23_layer(conv22)

        layers_toreverse = [conv22, conv19, conv16, conv13, conv9, conv7, conv5, conv3, conv1]
        all_switches = [switches_mask4, switches_mask3, switches_mask2, switches_mask1]
        conv_layers_down = [(conv23_layer, conv23), (conv22_layer, conv22), (conv21_layer, conv21), (conv20_layer, conv20),
                            (conv19_layer, conv19), (conv18_layer, conv18), (conv17_layer, conv17), (conv16_layer, conv16),
                            (conv15_layer, conv15), (conv14_layer, conv14), (conv13_layer, conv13), (conv12_layer, conv12),
                            (conv11_layer, conv11)]
        conv_layers_up = [(conv10_layer,conv10), (conv9_layer, conv9), (conv8_layer, conv8), (conv7_layer, conv7),
                          (conv6_layer, conv6), (conv5_layer, conv5), (conv4_layer, conv4), (conv3_layer, conv3),
                          (conv2_layer, conv2), (conv1_layer, conv1)]
        upsamp_layers = [up_samp4_layer, up_samp3_layer, up_samp2_layer, up_samp1_layer]

        reversed_layers = self.reverse_all_layers(deconv_inputs, layers_toreverse, all_switches, conv_layers_down, conv_layers_up, upsamp_layers)

        model = models.Model(inputs=inputs, outputs=conv23)

        deconv_models = self.generate_deconv_models(deconv_inputs, reversed_layers)

        model.compile(optimizer=Adam(lr=0.000001), loss=dice_coef_loss, metrics=[dice_coef, binary_crossentropy, weighted_crossentropy,
                                                                                   predicted_count, predicted_sum, ground_truth_count,
                                                                                 ground_truth_sum, recall])
        model.summary()

        return model, deconv_models

    def generate_deconv_models(self, deconv_inputs, reversed_layers):
        deconv_models = []
        for index, rev_layer in enumerate(reversed_layers):
            deconv_models.append(models.Model(inputs=deconv_inputs, outputs=rev_layer))
            deconv_models[index].summary()
        return deconv_models



    def reverse_all_layers(self, deconv_input,layers_to_reverse, all_switches,
                            conv_layers_down, conv_layers_up, upsamp_layers):

        reversed_layers = []
        for index, layer_tr in enumerate(layers_to_reverse):
            switches_list = all_switches[0: max(index - 4, 0)]
            filters_up = [K.int_shape(conv_layer[1])[-1] for conv_layer in conv_layers_up[0: max((index*2) - (4*2), 0)]]
            filters_down = [K.int_shape(conv_layer[1])[-1] for conv_layer in conv_layers_down[0:(index+1)*3]]
            kernel_sizes = [conv_layer[0].kernel_size for conv_layer in conv_layers_down[0:(index+1)*3] + conv_layers_up[0: max((index*2) - (4*2), 0)]]
            upsamp_size = [layer.size for layer in upsamp_layers[0:index+1]]
            reversed_layer = self.reverse_layer_path(deconv_input, switches_list, filters_up,
                                                     filters_down, kernel_sizes,
                                                     upsamp_size)
            reversed_layers.append(reversed_layer)

        return reversed_layers


    def reverse_layer_path(self, deconv_input, switches_list, filters_up,
                           filters_down, kernel_sizes, upsamp_sizes):

        k_size_counter = 0
        input_layer = layers.Conv2DTranspose(filters_down[0] + 1,
                                             kernel_size=kernel_sizes[k_size_counter][0],
                                             padding='same', kernel_initializer='he_normal',
                                             activation='relu', trainable=False)(deconv_input)
        k_size_counter += 1

        for index in range(1, len(filters_down)//3):

            input_layer = layers.Conv2DTranspose(filters_down[index],
                                                 kernel_size=kernel_sizes[k_size_counter],
                                                 padding='same', kernel_initializer='he_normal',
                                                 activation='relu', trainable=False)(input_layer)

            k_size_counter += 1
            input_layer = layers.Conv2DTranspose(filters_down[index + 1],
                                                 kernel_size=kernel_sizes[k_size_counter],
                                                 padding='same', kernel_initializer='he_normal',
                                                 activation='relu', trainable=False)(input_layer)
            k_size_counter += 1

            input_layer = layers.Conv2DTranspose(filters_down[index + 2],
                                                 kernel_size=kernel_sizes[k_size_counter],
                                                 padding='same', kernel_initializer='he_normal',
                                                 activation='relu', trainable=False)(input_layer)
            k_size_counter += 1

            input_layer = layers.Lambda(custom_layers.reverse_upconcat,
                                        arguments={
                                                   "height_factor": upsamp_sizes[index][0]/upsamp_sizes[index][0]**2,
                                                   "width_factor": upsamp_sizes[index][1]/upsamp_sizes[index][1]**2},
                                        output_shape=custom_layers.reverse_upconcat_output_shape(
                                            K.int_shape(input_layer),
                                            upsamp_sizes[index][0] /
                                            upsamp_sizes[index][0] ** 2,
                                            upsamp_sizes[index][1] /
                                            upsamp_sizes[index][1] ** 2)
                                        )(input_layer)

        for index, switches in enumerate(switches_list):

            input_layer = layers.Conv2DTranspose(filters_up[index], kernel_size=kernel_sizes[k_size_counter],
                                                 padding='same', kernel_initializer='he_normal',
                                                 activation='relu', trainable=False)(input_layer)
            k_size_counter += 1
            input_layer = layers.Conv2DTranspose(filters_up[index + 1], kernel_size=kernel_sizes[k_size_counter],
                                                 padding='same', kernel_initializer='he_normal',
                                                 activation='relu', trainable=False)(input_layer)
            k_size_counter += 1
            input_layer = layers.Lambda(custom_layers.unpooling_with_argmax2D,
                                        arguments={"poolsize": (2, 2), "argmax": switches},
                                        output_shape=K.int_shape(switches)[1:])(input_layer)


        if len(switches_list) == 4:
            input_layer = layers.Conv2DTranspose(filters_up[-2], kernel_size=kernel_sizes[k_size_counter],
                                                        padding='same', kernel_initializer='he_normal',
                                                        activation='relu', trainable=False)(input_layer)
            input_layer = layers.Conv2DTranspose(filters_up[-1], kernel_size=kernel_sizes[k_size_counter],
                                                 padding='same', kernel_initializer='he_normal',
                                                 activation='relu', trainable=False)(input_layer)


        return input_layer

    """
    def reverse_layer_path(self, input_layer, switches_list, filters_up,
                           filters_down, kernel_sizes, upsamp_sizes):

        k_size_counter = 0

        for index in range(0, len(filters_down)//3):

            input_layer = layers.Conv2DTranspose(filters_down[index],
                                                 kernel_size=kernel_sizes[k_size_counter],
                                                 padding='same', kernel_initializer='he_normal',
                                                 activation='relu')(input_layer)
            k_size_counter += 1
            input_layer = layers.Conv2DTranspose(filters_down[index + 1],
                                                 kernel_size=kernel_sizes[k_size_counter],
                                                 padding='same', kernel_initializer='he_normal',
                                                 activation='relu')(input_layer)
            k_size_counter += 1

            input_layer = layers.Conv2DTranspose(filters_down[index + 2],
                                                 kernel_size=kernel_sizes[k_size_counter],
                                                 padding='same', kernel_initializer='he_normal',
                                                 activation='relu')(input_layer)
            k_size_counter += 1

            input_layer = layers.Lambda(custom_layers.reverse_upconcat,
                                        arguments={
                                                   "height_factor": upsamp_sizes[index][0]/upsamp_sizes[index][0]**2,
                                                   "width_factor": upsamp_sizes[index][1]/upsamp_sizes[index][1]**2},
                                        output_shape=custom_layers.reverse_upconcat_output_shape(
                                            K.int_shape(input_layer),
                                            upsamp_sizes[index][0] /
                                            upsamp_sizes[index][0] ** 2,
                                            upsamp_sizes[index][1] /
                                            upsamp_sizes[index][1] ** 2)
                                        )(input_layer)


        for index, switches in enumerate(switches_list):

            input_layer = layers.Conv2DTranspose(filters_up[index], kernel_size=kernel_sizes[k_size_counter],
                                                 padding='same', kernel_initializer='he_normal',
                                                 activation='relu')(input_layer)
            k_size_counter += 1
            input_layer = layers.Conv2DTranspose(filters_up[index + 1], kernel_size=kernel_sizes[k_size_counter],
                                                 padding='same', kernel_initializer='he_normal',
                                                 activation='relu')(input_layer)
            k_size_counter += 1
            input_layer = layers.Lambda(custom_layers.unpooling_with_argmax2D,
                                      arguments={"poolsize": (2, 2), "argmax": switches},
                                      output_shape=K.int_shape(switches)[1:])(input_layer)


        second_last_deconv = layers.Conv2DTranspose(filters_up[-2], kernel_size=kernel_sizes[k_size_counter],
                                                    padding='same', kernel_initializer='he_normal',
                                                    activation='relu')(input_layer)
        last_deconv = layers.Conv2DTranspose(filters_up[-1], kernel_size=kernel_sizes[k_size_counter],
                                             padding='same', kernel_initializer='he_normal',
                                             activation='relu')(second_last_deconv)

        return last_deconv

    """

    def train(self, X, y, validation_data, training_name, base_path, epochs=10, batch_size=32):

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
            'batch_size': batch_size

        }
        self.save_specs(self.full_paths_dict['specs_path'], fit_specs)

        number_batches = self.save_batch_files(X, y, base_path, batch_size, "train")
        _ = self.save_batch_files(validation_data[0], validation_data[1], base_path, batch_size, "test")

        self.model.fit_generator(self.train_generator(base_path, number_batches, "train"),
                       steps_per_epoch=1,
                       callbacks=[checkpointer, tensorboard_callback],
                       epochs=epochs,
                       validation_steps=1,
                       validation_data=self.train_generator(base_path, number_batches, "test"),
                       verbose=1)


    def save_visualize_activations(self, data, labels, batch_size=1):

        selected_data, selected_labels = zip(*[[x, y] for x, y in sorted(zip(data, labels),
                                                                        key=lambda pair: len(np.nonzero(np.ravel(pair[1]))[0]),
                                                                        reverse=True)])
        selected_data, selected_labels = selected_data[:10], selected_labels[:10]
        viz_path = self.full_paths_dict['viz_path']

        for index, (original, label) in enumerate(zip(selected_data, selected_labels)):
            complete_path = self.create_viz_folders(viz_path, index, "_image")
            flair_t1 = np.concatenate([original[..., 0], original[..., 1]], axis=1)
            cv2.imwrite(os.path.join(complete_path, str(index) + "_original_deconv" + ".png"), flair_t1 * 255)
            cv2.imwrite(os.path.join(complete_path, str(index) + "_label_deconv" + ".png"), label * 255)
            label = np.expand_dims(label, 0)
            for layer_idx in range(len(self.deconv_models)):
                print("label shape", label.shape)
                self.deconv_models[layer_idx].summary()
                layer_pred = self.deconv_models[layer_idx].predict(label, batch_size=batch_size, verbose=1)
                layer_pred = np.squeeze(layer_pred, axis=0)
                layer_path = self.create_viz_folders(complete_path, layer_idx, "_layer")
                for channel in range(layer_pred.shape[-1]):
                    file_name = "deconv_activations_layer_chann_" + str(channel) + ".png"
                    full_path = os.path.join(layer_path, file_name)
                    cv2.imwrite(full_path, layer_pred[..., channel] * 255)



    def create_viz_folders(self, viz_path, index, suffix):

        complete_path = os.path.join(viz_path, str(index) + suffix)

        if not os.path.exists(complete_path):
            os.makedirs(complete_path)

        return complete_path