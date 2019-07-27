from augmentation.combineds import wassersteingan
import numpy as np
from preprocessing.imageparser import ImageParser
from constants import *
import gc
import os
import cv2

parser = ImageParser(path_utrech='../Utrecht/subjects',
                     path_singapore='../Singapore/subjects',
                     path_amsterdam='../GE3T/subjects')
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

rm_extra = 6
final_label_imgs = parser.preprocess_all_labels([labels_utrecht,
                                                 labels_singapore,
                                                 labels_amsterdam], slice_shape, [UTRECH_N_SLICES,
                                                                                  SINGAPORE_N_SLICES,
                                                                                  AMSTERDAM_N_SLICES])
'''

T1 DATA

'''
rm_total = (REMOVE_TOP + REMOVE_BOT) + 2 * rm_extra
utrecht_normalized_t1 = parser.preprocess_dataset_t1(t1_utrecht, slice_shape, UTRECH_N_SLICES,
                                                     REMOVE_TOP + rm_extra, REMOVE_BOT + rm_extra, norm_type="stand")
utrecht_normalized_t1 = parser.normalize_neg_pos_one(utrecht_normalized_t1, UTRECH_N_SLICES - rm_total)

singapore_normalized_t1 = parser.preprocess_dataset_t1(t1_singapore, slice_shape, SINGAPORE_N_SLICES,
                                                       REMOVE_TOP + rm_extra, REMOVE_BOT + rm_extra, norm_type="stand")
singapore_normalized_t1 = parser.normalize_neg_pos_one(singapore_normalized_t1, SINGAPORE_N_SLICES - rm_total)

amsterdam_normalized_t1 = parser.preprocess_dataset_t1(t1_amsterdam, slice_shape, AMSTERDAM_N_SLICES,
                                                       REMOVE_TOP + rm_extra, REMOVE_BOT + rm_extra, norm_type="stand")
amsterdam_normalized_t1 = parser.normalize_neg_pos_one(amsterdam_normalized_t1, AMSTERDAM_N_SLICES - rm_total)
del t1_utrecht, t1_singapore, t1_amsterdam

'''

FLAIR DATA

'''


utrecht_stand_flairs = parser.preprocess_dataset_flair(flair_utrecht, slice_shape, UTRECH_N_SLICES,
                                                       REMOVE_TOP + rm_extra, REMOVE_BOT + rm_extra, norm_type="stand")
utrecht_stand_flairs = parser.normalize_neg_pos_one(utrecht_stand_flairs, UTRECH_N_SLICES - rm_total)

singapore_stand_flairs = parser.preprocess_dataset_flair(flair_singapore, slice_shape, SINGAPORE_N_SLICES,
                                                       REMOVE_TOP + rm_extra, REMOVE_BOT + rm_extra, norm_type="stand")
singapore_stand_flairs = parser.normalize_neg_pos_one(singapore_stand_flairs, SINGAPORE_N_SLICES - rm_total)

amsterdam_stand_flairs = parser.preprocess_dataset_flair(flair_amsterdam, slice_shape, AMSTERDAM_N_SLICES,
                                                       REMOVE_TOP + rm_extra, REMOVE_BOT + rm_extra, norm_type="stand")
amsterdam_stand_flairs = parser.normalize_neg_pos_one(amsterdam_stand_flairs, AMSTERDAM_N_SLICES - rm_total)

del flair_utrecht, flair_singapore, flair_amsterdam


'''

DATA CONCAT

'''
normalized_t1 = np.concatenate([utrecht_normalized_t1,
                                singapore_normalized_t1,
                                amsterdam_normalized_t1], axis=0)

normalized_flairs = np.concatenate([utrecht_stand_flairs,
                                    singapore_stand_flairs,
                                    amsterdam_stand_flairs], axis=0)

del utrecht_normalized_t1, singapore_normalized_t1, amsterdam_normalized_t1
del utrecht_stand_flairs, singapore_stand_flairs, amsterdam_stand_flairs

data_t1 = np.expand_dims(np.asanyarray(normalized_t1), axis=3)
data_flair = np.expand_dims(np.asanyarray(normalized_flairs), axis=3)
all_data = np.concatenate([data_t1, data_flair], axis=3)

del data_t1, data_flair

gc.collect()

training_name = "wasserstein_gan_test1_v1"
base_path = os.getcwd()
print("HEREEEE ", (*all_data.shape[1:-1], all_data.shape[-1]))
GAN = wassersteingan.WassersteinGAN(img_shape=(*all_data.shape[1:-1], all_data.shape[-1]), noise_shape=(128,))
GAN.train(all_data, base_path=base_path, training_name=training_name,
          epochs=5000, batch_size=64, save_interval=50)
