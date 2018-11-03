from keras import models
from keras import layers
from contextlib import redirect_stdout
import os
from keras.callbacks import ModelCheckpoint, TensorBoard
from sklearn.model_selection import train_test_split
from keras.models import load_model
from keras.optimizers import Adam, SGD
from metrics import dice_coef, dice_coef_loss, weighted_crossentropy, \
    predicted_count, ground_truth_count, ground_truth_sum, predicted_sum
import custom_layers
from imageparser import ImageParser
import numpy as np
import keras.backend as K


parser = ImageParser()
utrech_dataset, _, _ = parser.get_all_images_and_labels()

slice_shape = (240, 240)

t1_utrecht = [row[1] for row in utrech_dataset]
flair_utrecht = [row[2] for row in utrech_dataset]
labels_utrecht = [row[0] for row in utrech_dataset]

utrecht_data_t1 = parser.get_all_images_np_twod(t1_utrecht)
utrecht_resized_t1 = parser.resize_slices(utrecht_data_t1, slice_shape)
utrecht_normalized_t1 = parser.normalize_images(utrecht_resized_t1)
utrecht_normalized_t1 = np.expand_dims(np.asanyarray(utrecht_normalized_t1), 3)

utrecht_data_flair = parser.get_all_images_np_twod(flair_utrecht)
utrecht_resized_flairs = parser.resize_slices(utrecht_data_flair, slice_shape)
utrecht_normalized_flairs = parser.normalize_images(utrecht_resized_flairs)
utrecht_normalized_flairs = np.expand_dims(np.asanyarray(utrecht_normalized_flairs), 3)

print("T1 and FLAIR shapes: ", utrecht_normalized_t1.shape, utrecht_normalized_flairs.shape)

all_data = np.concatenate([utrecht_normalized_t1, utrecht_normalized_flairs], axis=3)
print("Data shape: ", all_data.shape)

inputs = layers.Input(shape=all_data.shape[1:])
conv1 = layers.Conv2D(64, kernel_size=5, padding='same',
                      kernel_initializer='he_normal', activation='relu')(inputs)
conv2 = layers.Conv2D(64, kernel_size=5, padding='same',
                      kernel_initializer='he_normal', activation='relu')(conv1)

pooling1, switches_mask = custom_layers.MaxPoolingWithArgmax2D(pool_size=(2, 2))(conv2)

unpooled1 = layers.Lambda(custom_layers.unpooling_with_argmax2D,
                            arguments={"poolsize": (2, 2), "argmax": switches_mask},
                            output_shape=custom_layers.unpoolingMask2D_output_shape)(pooling1)

conv_trans1 = layers.Conv2DTranspose(64, kernel_size=5, padding='same',
                                     kernel_initializer='he_normal', activation='relu')(unpooled1)
conv_trans2 = layers.Conv2DTranspose(1, kernel_size=5, padding='same',
                                     kernel_initializer='he_normal', activation='relu')(conv_trans1)


model = models.Model(inputs=[inputs], outputs=[conv_trans2])

model.compile(optimizer=Adam(lr=0.000001), loss=dice_coef_loss, metrics=[dice_coef,
                                                                         weighted_crossentropy,
                                                                         predicted_count,
                                                                         predicted_sum,
                                                                         ground_truth_count,
                                                                                 ground_truth_sum])

model.summary()