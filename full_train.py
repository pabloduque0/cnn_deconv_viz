import numpy as np
from networks.attention_unet import AttentionUnet
from preprocessing.imageparser import ImageParser
from augmentation.imageaugmentator import ImageAugmentator
from sklearn.model_selection import train_test_split
from constants import *
import gc
import os
import tensorflow as tf
import utils
import cv2

parser = ImageParser(path_utrech='../new_data_set/Utrecht/subjects',
                     path_singapore='../new_data_set/Singapore/subjects',
                     path_amsterdam='../new_data_set/GE3T/subjects')
utrech_dataset, singapore_dataset, amsterdam_dataset = parser.get_all_images_and_labels()

t1_utrecht, flair_utrecht, labels_utrecht, brain_mask_utrecht, distance_utrecht = parser.get_all_sets_paths(
    utrech_dataset)
t1_singapore, flair_singapore, labels_singapore, brain_mask_singapore, distance_singapore = parser.get_all_sets_paths(
    singapore_dataset)
t1_amsterdam, flair_amsterdam, labels_amsterdam, brain_mask_amsterdam, distance_amsterdam = parser.get_all_sets_paths(
    amsterdam_dataset)

slice_shape = SLICE_SHAPE

print('Utrecht: ', len(t1_utrecht), len(flair_utrecht), len(labels_utrecht))
print('Singapore: ', len(t1_singapore), len(flair_singapore), len(labels_singapore))
print('Amsterdam: ', len(t1_amsterdam), len(flair_amsterdam), len(labels_amsterdam))

"""

LABELS DATA

"""

labels_utrecht = parser.preprocess_all_labels([labels_utrecht], slice_shape, [UTRECH_N_SLICES],
                                              remove_top=REMOVE_TOP,
                                              remove_bot=REMOVE_BOT,
                                              rm_extra_amsterdam=(0, 0))

labels_singapore = parser.preprocess_all_labels([labels_singapore], slice_shape, [SINGAPORE_N_SLICES],
                                                remove_top=REMOVE_TOP,
                                                remove_bot=REMOVE_BOT,
                                                rm_extra_amsterdam=(0, 0))

labels_amsterdam = parser.preprocess_all_labels([labels_amsterdam], slice_shape, [AMSTERDAM_N_SLICES],
                                                remove_top=REMOVE_TOP,
                                                remove_bot=REMOVE_BOT,
                                                rm_extra_amsterdam=(0, 0))

labels = np.concatenate([labels_utrecht, labels_singapore, labels_amsterdam], axis=0)



'''

T1 DATA

'''
utrecht_normalized_t1 = parser.preprocess_dataset_t1(t1_utrecht, slice_shape, UTRECH_N_SLICES,
                                                     REMOVE_TOP, REMOVE_BOT)
singapore_normalized_t1 = parser.preprocess_dataset_t1(t1_singapore, slice_shape, SINGAPORE_N_SLICES,
                                                       REMOVE_TOP, REMOVE_BOT)
amsterdam_normalized_t1 = parser.preprocess_dataset_t1(t1_amsterdam, slice_shape, AMSTERDAM_N_SLICES,
                                                       REMOVE_TOP, REMOVE_BOT)

del t1_utrecht, t1_singapore, t1_amsterdam

'''

FLAIR DATA

'''

utrecht_stand_flairs = parser.preprocess_dataset_flair(flair_utrecht, slice_shape, UTRECH_N_SLICES,
                                                       REMOVE_TOP, REMOVE_BOT)
singapore_stand_flairs = parser.preprocess_dataset_flair(flair_singapore, slice_shape, SINGAPORE_N_SLICES,
                                                         REMOVE_TOP, REMOVE_BOT)
amsterdam_stand_flairs = parser.preprocess_dataset_flair(flair_amsterdam, slice_shape, AMSTERDAM_N_SLICES,
                                                         REMOVE_TOP, REMOVE_BOT)

del flair_utrecht, flair_singapore, flair_amsterdam

'''

DATA CONCAT

'''


t1_data = np.concatenate([utrecht_normalized_t1, singapore_normalized_t1, amsterdam_normalized_t1], axis=0)
flair_data = np.concatenate([utrecht_stand_flairs, singapore_stand_flairs, amsterdam_stand_flairs], axis=0)

train_data = np.concatenate([np.expand_dims(t1_data, axis=-1),
                             np.expand_dims(flair_data, axis=-1)], axis=3)

gc.collect()
'''

AUGMENTATION

'''
print(train_data.shape, labels.shape)
augmentator = ImageAugmentator()
data_augmented, labels_agumented = augmentator.perform_all_augmentations(train_data, labels)

data_train = np.asanyarray(data_augmented)
labels_train = np.asanyarray(labels_agumented)
del data_augmented, labels_agumented


for i in range(0, 3):

    '''

    TRAINING

    '''
    idx_swap = np.arange(data_train.shape[0])
    np.random.shuffle(idx_swap)

    data_train = data_train[idx_swap]
    labels_train = labels_train[idx_swap]

    training_name = '20200326_longtrain_ksize11_ensemb_3_{}_new'.format(i)
    base_path = os.getcwd()

    print(data_train.shape, labels_train.shape)
    unet = AttentionUnet(img_shape=data_train.shape[1:])

    unet.train(data_train, labels_train, None, training_name, base_path, epochs=35, batch_size=15)

