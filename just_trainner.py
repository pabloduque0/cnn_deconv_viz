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

'''

AUGMENTATION


'''

all_data = np.load("data.npy")
final_label_imgs = np.load("labels.npy")

augmentator = ImageAugmentator()
data_augmented, labels_agumented = augmentator.perform_all_augmentations(all_data, final_label_imgs)

data_train, validation_data, labels_train, validation_labels = train_test_split(data_augmented, labels_agumented, test_size=0.04)

del data_augmented, labels_agumented
data_train = np.asanyarray(data_train)
labels_train = np.asanyarray(labels_train)

'''

TRAINING

'''
gc.collect()
print(data_train.dtype, labels_train.dtype)
training_name = '6_adaptative_normalization'
base_path = '/harddrive/home/pablo/Google Drive/UNED/TFM/cnn_deconv_viz'
test_size = 0.15

print(data_train.shape, labels_train.shape)
unet = OldUnet(img_shape=data_train.shape[1:])
unet.train(data_train, labels_train, test_size, training_name, base_path, epochs=40, batch_size=30)

'''

VALIDATING

'''

validation_data = np.asanyarray(validation_data)
validation_labels = np.asanyarray(validation_labels)
unet.predict_and_save(validation_data, validation_labels)

"""

VISUALIZING

"""

#unet.visualize_activations(data_train, labels_train, batch_size=30)
