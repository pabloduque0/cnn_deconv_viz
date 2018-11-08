import numpy as np
from unetdeconv import UnetDeconv
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
utrecht_normalized_t1 = parser.normalize_images(utrecht_resized_t1)

del utrecht_data_t1, utrecht_resized_t1, t1_utrecht

singapore_data_t1 = parser.get_all_images_np_twod(t1_singapore)
singapore_resized_t1 = parser.resize_slices(singapore_data_t1, slice_shape)
singapore_normalized_t1 = parser.normalize_images(singapore_resized_t1)

del singapore_data_t1, singapore_resized_t1, t1_singapore

amsterdam_data_t1 = parser.get_all_images_np_twod(t1_amsterdam)
amsterdam_resized_t1 = parser.resize_slices(amsterdam_data_t1, slice_shape)
amsterdam_normalized_t1 = parser.normalize_images(amsterdam_resized_t1)

del amsterdam_data_t1, amsterdam_resized_t1, t1_amsterdam

#print('T1: ', np.max(np.asanyarray(utrecht_data_t1).ravel()), np.max(np.asanyarray(singapore_resized_t1).ravel()), np.max(np.asanyarray(amsterdam_data_t1).ravel()))

'''

FLAIR DATA

'''

utrecht_data_flair = parser.get_all_images_np_twod(flair_utrecht)
utrecht_resized_flairs = parser.resize_slices(utrecht_data_flair, slice_shape)
utrecht_normalized_flairs = parser.normalize_images(utrecht_resized_flairs)

del utrecht_data_flair, utrecht_resized_flairs, flair_utrecht

utrecht_data_tophat = parser.generate_tophat(utrecht_normalized_flairs)

singapore_data_flair = parser.get_all_images_np_twod(flair_singapore)
singapore_resized_flairs = parser.resize_slices(singapore_data_flair, slice_shape)
singapore_normalized_flairs = parser.normalize_images(singapore_resized_flairs)

del singapore_data_flair, singapore_resized_flairs, flair_singapore

singapore_data_tophat = parser.generate_tophat(singapore_normalized_flairs)

amsterdam_data_flair = parser.get_all_images_np_twod(flair_amsterdam)
amsterdam_resized_flairs = parser.resize_slices(amsterdam_data_flair, slice_shape)
amsterdam_normalized_flairs = parser.normalize_images(amsterdam_resized_flairs)

del amsterdam_data_flair, amsterdam_resized_flairs, flair_amsterdam

amsterdam_data_tophat = parser.generate_tophat(amsterdam_normalized_flairs)

#print('Flairs: ', np.max(np.asanyarray(utrecht_data_flair)), np.max(np.asanyarray(singapore_resized_flairs)), np.max(np.asanyarray(amsterdam_data_flair)))

'''

DATA CONCAT

'''

normalized_t1 = utrecht_normalized_t1 + singapore_normalized_t1 + amsterdam_normalized_t1
normalized_flairs = utrecht_normalized_flairs + singapore_normalized_flairs + amsterdam_normalized_flairs
data_tophat = utrecht_data_tophat + singapore_data_tophat + amsterdam_data_tophat

del utrecht_normalized_t1, singapore_normalized_t1, amsterdam_normalized_t1
del utrecht_normalized_flairs, singapore_normalized_flairs, amsterdam_normalized_flairs
del utrecht_data_tophat, singapore_data_tophat, amsterdam_data_tophat

data_t1 = np.expand_dims(np.asanyarray(normalized_t1), axis=3)
data_flair = np.expand_dims(np.asanyarray(normalized_flairs), axis=3)
data_tophat = np.asanyarray(data_tophat)

all_data = np.concatenate([data_t1, data_flair, data_tophat], axis=3)

del data_t1
del data_flair
del data_tophat

# All labels as np
all_labels_paths = labels_utrecht + labels_singapore + labels_amsterdam
all_labels_imgs = parser.get_all_images_np_twod(all_labels_paths)

"""
# Extra gets for separated analysis
labels_utrecht_imgs = parser.get_all_images_np_twod(labels_utrecht)
labels_singapore_imgs = parser.get_all_images_np_twod(labels_singapore)
labels_amsterdam_imgs = parser.get_all_images_np_twod(labels_amsterdam)
"""

resized_labels = parser.resize_slices(all_labels_imgs, slice_shape)
final_label_imgs = parser.remove_third_label(resized_labels)

del all_labels_imgs, resized_labels, labels_utrecht, labels_singapore, labels_amsterdam


print('Total pixels: ', len(np.ravel(np.asanyarray(final_label_imgs))))
print('White pixels: ', len(np.flatnonzero(np.asanyarray(final_label_imgs))))

final_label_imgs = np.expand_dims(np.asanyarray(final_label_imgs), axis=3)

gc.collect()
'''

AUGMENTATION

'''
print("Size of one image: ", sys.getsizeof([0]))
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

output_path = base_path + '/output/'
validation_data = np.asanyarray(validation_data)
validation_labels = np.asanyarray(validation_labels)

unet.predict_and_save(validation_data, validation_labels, output_path)

"""

VISUALIZING

"""

unet.visualize_activations(data_train, labels_train, "/viz_output/", batch_size=30)
