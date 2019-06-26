import numpy as np
from networks.ustepnet import UStepNet
from networks.unet import Unet
from networks.shallowunet import ShallowUnet
from networks.stacknet import StackNet
from preprocessing.imageparser import ImageParser
from augmentation.imageaugmentator import ImageAugmentator
from sklearn.model_selection import train_test_split
from constants import *
import gc
import os
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)

parser = ImageParser()
utrech_dataset, singapore_dataset, amsterdam_dataset = parser.get_all_images_and_labels()

t1_utrecht, flair_utrecht, labels_utrecht, white_mask_utrecht, distance_utrecht = parser.get_all_sets_paths(utrech_dataset)
t1_singapore, flair_singapore, labels_singapore, white_mask_singapore, distance_singapore = parser.get_all_sets_paths(singapore_dataset)
t1_amsterdam, flair_amsterdam, labels_amsterdam, white_mask_amsterdam, distance_amsterdam = parser.get_all_sets_paths(amsterdam_dataset)

slice_shape = SLICE_SHAPE

print('Utrecht: ', len(t1_utrecht), len(flair_utrecht), len(labels_utrecht))
print('Singapore: ', len(t1_singapore), len(flair_singapore), len(labels_singapore))
print('Amsterdam: ', len(t1_amsterdam), len(flair_amsterdam), len(labels_amsterdam))


"""

LABELS DATA

"""


final_label_imgs = parser.preprocess_all_labels([labels_utrecht,
                                                 labels_singapore,
                                                 labels_amsterdam], slice_shape)

'''

T1 DATA

'''
utrecht_data_t1 = parser.get_all_images_np_twod(t1_utrecht)
utrecht_resized_t1 = parser.resize_slices(utrecht_data_t1, slice_shape)
utrecht_resized_t1 = parser.remove_top_bot_slices(utrecht_resized_t1, UTRECH_N_SLICES)
utrecht_normalized_t1 = parser.standarize(utrecht_resized_t1,
                                                UTRECH_N_SLICES - REMOVE_TOP - REMOVE_BOT)

del utrecht_data_t1, utrecht_resized_t1, t1_utrecht

singapore_data_t1 = parser.get_all_images_np_twod(t1_singapore)
singapore_resized_t1 = parser.resize_slices(singapore_data_t1, slice_shape)
singapore_resized_t1 = parser.remove_top_bot_slices(singapore_resized_t1, SINGAPORE_N_SLICES)
singapore_normalized_t1 = parser.standarize(singapore_resized_t1,
                                                  SINGAPORE_N_SLICES - REMOVE_TOP - REMOVE_BOT)

del singapore_data_t1, singapore_resized_t1, t1_singapore

amsterdam_data_t1 = parser.get_all_images_np_twod(t1_amsterdam)
amsterdam_resized_t1 = parser.resize_slices(amsterdam_data_t1, slice_shape)
amsterdam_resized_t1 = parser.remove_top_bot_slices(amsterdam_resized_t1, AMSTERDAM_N_SLICES)
amsterdam_normalized_t1 = parser.standarize(amsterdam_resized_t1,
                                                  AMSTERDAM_N_SLICES - REMOVE_TOP - REMOVE_BOT)

del amsterdam_data_t1, amsterdam_resized_t1, t1_amsterdam


'''

FLAIR DATA

'''

utrecht_data_flair = parser.get_all_images_np_twod(flair_utrecht)
utrecht_resized_flairs = parser.resize_slices(utrecht_data_flair, slice_shape)
utrecht_resized_flairs = parser.remove_top_bot_slices(utrecht_resized_flairs, UTRECH_N_SLICES)
utrecht_normalized_flairs = parser.standarize(utrecht_resized_flairs,
                                                      UTRECH_N_SLICES - REMOVE_TOP - REMOVE_BOT)

del utrecht_data_flair, utrecht_resized_flairs, flair_utrecht

singapore_data_flair = parser.get_all_images_np_twod(flair_singapore)
singapore_resized_flairs = parser.resize_slices(singapore_data_flair, slice_shape)
singapore_resized_flairs = parser.remove_top_bot_slices(singapore_resized_flairs, SINGAPORE_N_SLICES)
singapore_normalized_flairs = parser.standarize(singapore_resized_flairs,
                                                        SINGAPORE_N_SLICES - REMOVE_TOP - REMOVE_BOT)

del singapore_data_flair, singapore_resized_flairs, flair_singapore

amsterdam_data_flair = parser.get_all_images_np_twod(flair_amsterdam)
amsterdam_resized_flairs = parser.resize_slices(amsterdam_data_flair, slice_shape)
amsterdam_resized_flairs = parser.remove_top_bot_slices(amsterdam_resized_flairs, AMSTERDAM_N_SLICES)
amsterdam_normalized_flairs = parser.standarize(amsterdam_resized_flairs,
                                                        AMSTERDAM_N_SLICES - REMOVE_TOP - REMOVE_BOT)


del amsterdam_data_flair, amsterdam_resized_flairs, flair_amsterdam


'''

DATA CONCAT

'''
normalized_t1 = np.concatenate([utrecht_normalized_t1,
                                singapore_normalized_t1,
                                amsterdam_normalized_t1], axis=0)
normalized_flairs = np.concatenate([utrecht_normalized_flairs,
                                    singapore_normalized_flairs,
                                    amsterdam_normalized_flairs], axis=0)


del utrecht_normalized_t1, singapore_normalized_t1, amsterdam_normalized_t1
del utrecht_normalized_flairs, singapore_normalized_flairs, amsterdam_normalized_flairs

data_t1 = np.expand_dims(np.asanyarray(normalized_t1), axis=3)
data_flair = np.expand_dims(np.asanyarray(normalized_flairs), axis=3)
all_data = np.concatenate([data_t1, data_flair], axis=3)



del data_t1, data_flair

gc.collect()

'''

AUGMENTATION

'''
data_train, test_data, labels_train, test_labels = train_test_split(all_data, final_label_imgs,
                                                                    test_size=0.15, random_state=42)

data_train, validation_data, labels_train, validation_labels = train_test_split(data_train, labels_train,
                                                                                test_size=0.04, random_state=42)

augmentator = ImageAugmentator()
data_augmented, labels_agumented = augmentator.perform_all_augmentations(data_train, labels_train)

data_train = np.asanyarray(data_augmented)
labels_train = np.asanyarray(labels_agumented)
del data_augmented, labels_agumented
'''

TRAINING

'''
gc.collect()
training_name = 'test_new_deconv'
base_path = os.getcwd()

print(data_train.shape, labels_train.shape, test_data.shape, test_labels.shape)
unet = ShallowUnet(img_shape=data_train.shape[1:])
unet.train(data_train, labels_train, (test_data, test_labels), training_name, base_path, epochs=30, batch_size=30)


'''

VALIDATING

'''
validation_data = np.asanyarray(validation_data)
validation_labels = np.asanyarray(validation_labels)
unet.predict_and_save(validation_data, validation_labels)

"""

VISUALIZING

"""
del data_train, labels_train, training_name
gc.collect()
#unet.save_visualize_activations(validation_data, validation_labels, batch_size=30)
