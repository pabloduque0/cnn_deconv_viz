import numpy as np
from unetdeconv import UnetDeconv
from unet import Unet
from oldunet import OldUnet
from imageparser import ImageParser
from imageaugmentator import ImageAugmentator
from sklearn.model_selection import train_test_split
import cv2
import sys
import gc


parser = ImageParser()
utrech_dataset, singapore_dataset, amsterdam_dataset = parser.get_all_images_and_labels()

t1_utrecht = [row[1] for row in utrech_dataset]
flair_utrecht = [row[2] for row in utrech_dataset]
labels_utrecht = [row[0] for row in utrech_dataset]

t1_singapore = [row[1] for row in singapore_dataset]
flair_singapore = [row[2] for row in singapore_dataset]
labels_singapore = [row[0] for row in singapore_dataset]

t1_amsterdam = [row[1] for row in amsterdam_dataset]
flair_amsterdam = [row[2] for row in amsterdam_dataset]
labels_amsterdam = [row[0] for row in amsterdam_dataset]

slice_shape = (240, 240)

print('Utrecht: ', len(t1_utrecht), len(flair_utrecht), len(labels_utrecht))
print('Singapore: ', len(t1_singapore), len(flair_singapore), len(labels_singapore))
print('Amsterdam: ', len(t1_amsterdam), len(flair_amsterdam), len(labels_amsterdam))

'''

T1 DATA

'''
utrecht_data_t1 = parser.get_all_images_np_twod(t1_utrecht)
utrecht_resized_t1 = parser.resize_slices(utrecht_data_t1, slice_shape)
utrecht_normalized_t1 = parser.normalize_minmax(utrecht_resized_t1, 48)

del utrecht_data_t1, utrecht_resized_t1, t1_utrecht

singapore_data_t1 = parser.get_all_images_np_twod(t1_singapore)
singapore_resized_t1 = parser.resize_slices(singapore_data_t1, slice_shape)
singapore_normalized_t1 = parser.normalize_minmax(singapore_resized_t1, 48)

del singapore_data_t1, singapore_resized_t1, t1_singapore

amsterdam_data_t1 = parser.get_all_images_np_twod(t1_amsterdam)
amsterdam_resized_t1 = parser.resize_slices(amsterdam_data_t1, slice_shape)
amsterdam_normalized_t1 = parser.normalize_minmax(amsterdam_resized_t1, 83)

del amsterdam_data_t1, amsterdam_resized_t1, t1_amsterdam

"""

LABELS DATA

"""

labels_utrecht_imgs = parser.get_all_images_np_twod(labels_utrecht)
labels_singapore_imgs = parser.get_all_images_np_twod(labels_singapore)
labels_amsterdam_imgs = parser.get_all_images_np_twod(labels_amsterdam)

labels_utrecht_resized = parser.resize_slices(labels_utrecht_imgs, slice_shape)
labels_singapore_resized = parser.resize_slices(labels_singapore_imgs, slice_shape)
labels_amsterdam_resized = parser.resize_slices(labels_amsterdam_imgs, slice_shape)

labels_utrecht_resized = parser.remove_third_label(labels_utrecht_resized)
labels_singapore_resized = parser.remove_third_label(labels_singapore_resized)
labels_amsterdam_resized = parser.remove_third_label(labels_amsterdam_resized)

final_label_imgs = np.concatenate([labels_utrecht_resized,
                                   labels_singapore_resized,
                                   labels_amsterdam_resized], axis=0)
final_label_imgs = np.expand_dims(np.asanyarray(final_label_imgs), axis=3)

del labels_utrecht_imgs, labels_singapore_imgs, labels_amsterdam_imgs



'''

FLAIR DATA

'''

utrecht_data_flair = parser.get_all_images_np_twod(flair_utrecht)
utrecht_resized_flairs = parser.resize_slices(utrecht_data_flair, slice_shape)
utrecht_normalized_flairs = parser.normalize_quantile(utrecht_resized_flairs, labels_utrecht_resized, 48)

del utrecht_data_flair, utrecht_resized_flairs, flair_utrecht

singapore_data_flair = parser.get_all_images_np_twod(flair_singapore)
singapore_resized_flairs = parser.resize_slices(singapore_data_flair, slice_shape)
singapore_normalized_flairs = parser.normalize_quantile(singapore_resized_flairs, labels_singapore_resized, 48)

del singapore_data_flair, singapore_resized_flairs, flair_singapore

amsterdam_data_flair = parser.get_all_images_np_twod(flair_amsterdam)
amsterdam_resized_flairs = parser.resize_slices(amsterdam_data_flair, slice_shape)
amsterdam_normalized_flairs = parser.normalize_quantile(amsterdam_resized_flairs, labels_amsterdam_resized, 83)

del amsterdam_data_flair, amsterdam_resized_flairs, flair_amsterdam
del labels_utrecht_resized, labels_singapore_resized, labels_amsterdam_resized
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

del data_t1
del data_flair



gc.collect()
'''

AUGMENTATION


'''
augmentator = ImageAugmentator()
data_augmented, labels_agumented = augmentator.perform_all_augmentations(all_data, final_label_imgs)

data_train, validation_data, labels_train, validation_labels = train_test_split(data_augmented, labels_agumented, test_size=0.05)

del data_augmented, labels_agumented
data_train = np.asanyarray(data_train)
labels_train = np.asanyarray(labels_train)

'''

TRAINING

'''
gc.collect()

training_name = 'deconv_test1'
base_path = '/harddrive/home/pablo/Google Drive/UNED/TFM/cnn_deconv_viz'
test_size = 0.3

print(data_train.shape, labels_train.shape)

unet = UnetDeconv(img_shape=data_train.shape[1:])
unet.train_with_generator(data_train, labels_train, test_size, training_name, base_path, epochs=1, batch_size=30)

'''

VALIDATING

'''

validation_data = np.asanyarray(validation_data)
validation_labels = np.asanyarray(validation_labels)
unet.predict_and_save(validation_data, validation_labels)

"""

VISUALIZING

"""
del data_train, labels_train, test_size, training_name
gc.collect()
unet.visualize_activations(validation_data, validation_labels, batch_size=30)
