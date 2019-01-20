#############################################################################
#                                                                           #
# BASED KERAS IMAGE AUGMENTATION:                                           #
# https://keras.io/preprocessing/image/                                     #
#                                                                           #
#############################################################################

import numpy as np
import scipy.ndimage as ndi
import gc
from keras.preprocessing.image import apply_affine_transform
import random
import cv2

class ImageAugmentator():

    def __init__(self, theta_range=(-30, 30), x_shift_range=(-.3, .3), y_shift_range=(-.3, .3),
                 zx_range=(0.9, 1.2), zy_range=(0.9, 1.2), shear_range=(-.2, .2)):
        self.theta_range = theta_range
        self.x_shift_range = x_shift_range
        self.y_shift_range = y_shift_range
        self.zx_range = zx_range
        self.zy_range = zy_range
        self.shear_range = shear_range

    def perform_all_augmentations(self, dataset_x, dataset_y, visualize=False):

        if len(dataset_x) != len(dataset_y):
            raise ValueError("Wrong input. Image lists must be have the same length.")

        non_black_indices = [index for index, image in enumerate(dataset_y) if image[image > 0].shape != (0,)]
        idx_group1, idx_group2, idx_group3, idx_group4, idx_group5 = self.make_indices_groups(non_black_indices, 5)
        # Rotations
        rotated_xs, rotated_ys = self.perform_rotations(dataset_x[idx_group1],
                                                        dataset_y[idx_group1])
        aug_dataset_x = np.concatenate([dataset_x, rotated_xs], axis=0)
        rotated_ys = np.expand_dims(np.asanyarray(rotated_ys)[:, :, :, 0], axis=3)
        aug_dataset_y = np.concatenate([dataset_y, rotated_ys], axis=0)
        del rotated_xs, rotated_ys

        # Shifts
        shifted_xs, shifted_ys = self.perform_shifts(dataset_x[idx_group2],
                                                     dataset_y[idx_group2])
        aug_dataset_x = np.concatenate([aug_dataset_x, shifted_xs], axis=0)
        aug_dataset_y = np.concatenate([aug_dataset_y, shifted_ys], axis=0)
        del shifted_xs, shifted_ys

        # Shear
        sheared_xs, sheared_ys = self.perform_shears(dataset_x[idx_group3],
                                                     dataset_y[idx_group3])
        aug_dataset_x = np.concatenate([aug_dataset_x, sheared_xs], axis=0)
        aug_dataset_y = np.concatenate([aug_dataset_y, sheared_ys], axis=0)
        del sheared_xs, sheared_ys

        # Zoom
        zoomed_xs, zoomed_ys = self.apply_zoom(dataset_x[idx_group4],
                                               dataset_y[idx_group4])
        aug_dataset_x = np.concatenate([aug_dataset_x, zoomed_xs], axis=0)
        aug_dataset_y = np.concatenate([aug_dataset_y, zoomed_ys], axis=0)
        del zoomed_xs, zoomed_ys

        # Flips
        flipped_xs, flipped_ys = self.perform_flips(dataset_x[idx_group5],
                                                    dataset_y[idx_group5])
        aug_dataset_x = np.concatenate([aug_dataset_x, flipped_xs], axis=0)
        aug_dataset_y = np.concatenate([aug_dataset_y, flipped_ys], axis=0)
        del flipped_xs, flipped_ys


        # Multiple augmentations
        mult_xs, mult_ys = self.mutiple_agumentations(dataset_x[non_black_indices], dataset_y[non_black_indices])
        aug_dataset_x = np.concatenate([aug_dataset_x, mult_xs], axis=0)
        aug_dataset_y = np.concatenate([aug_dataset_y, mult_ys], axis=0)
        del mult_xs, mult_ys
        del idx_group1, idx_group2, idx_group3, idx_group4, idx_group5
        gc.collect()

        mixedup_x, mixedup_y = self.apply_mixup(non_black_indices, dataset_x, dataset_y)
        aug_dataset_x = np.concatenate([aug_dataset_x, mixedup_x], axis=0)
        aug_dataset_y = np.concatenate([aug_dataset_y, mixedup_y], axis=0)

        if visualize:
            self.visualize_data_augmentation(dataset_x[non_black_indices],
                                         dataset_y[non_black_indices],
                                     aug_dataset_x[len(dataset_x):, ...],
                                     aug_dataset_y[len(dataset_y):, ...])

        return aug_dataset_x, aug_dataset_y

    def apply_mixup(self, non_black_indices, images_x, images_y, alpha=0.4):

        idx_group1, idx_group2 = self.make_indices_groups(non_black_indices, 2)
        if len(idx_group1) > len(idx_group2): del idx_group1[-1]
        if len(idx_group2) > len(idx_group1): del idx_group2[-1]

        output_x = images_x[idx_group1] * alpha + (1 - alpha) * images_x[idx_group2]
        output_y = images_y[idx_group1] * alpha + (1 - alpha) * images_y[idx_group2]

        return output_x, output_y


    def make_indices_groups(self, indices, n_groups):

        size_group = len(indices) // n_groups
        list_groups = []
        for i in range(n_groups):

            if i < n_groups - 1:
                group = indices[i * size_group : (i+1) * size_group]
            else:
                group = indices[i * size_group: ]

            list_groups.append(sorted(list(group)))

        return list_groups


    def sample_indices_groups(self, indices, n_groups):
        size_group = len(indices) // n_groups
        list_groups = []
        for i in range(n_groups):
            group = random.sample(indices, size_group)
            list_groups.append(sorted(list(group)))
            indices = sorted(list(set(indices).difference(set(group))))

        return list_groups

    def mutiple_agumentations(self, images_x, images_y):

        if len(images_x) != len(images_y):
            raise ValueError("Wrong input. Image lists must be have the same length.")

        augmented_list_x = []
        augmented_list_y = []
        for image_x, image_y in zip(images_x, images_y):
            angle = np.random.uniform(*self.theta_range)
            width_shift = np.random.uniform(*self.x_shift_range)
            height_shift = np.random.uniform(*self.y_shift_range)
            shear = np.random.uniform(*self.shear_range)
            zoom_x_value = np.random.uniform(*self.zx_range)
            zoom_y_value = np.random.uniform(*self.zy_range)

            augmented_x = apply_affine_transform(image_x,
                                                 theta=angle,
                                                 tx=width_shift,
                                                 ty=height_shift,
                                                 shear=shear,
                                                 zx=zoom_x_value,
                                                 zy=zoom_y_value,
                                                 fill_mode="constant", cval=0.)
            agumented_y = apply_affine_transform(image_y,
                                                 theta=angle,
                                                 tx=width_shift,
                                                 ty=height_shift,
                                                 shear=shear,
                                                 zx=zoom_x_value,
                                                 zy=zoom_y_value,
                                                 fill_mode="constant", cval=0.)
            augmented_list_x.append(augmented_x)
            augmented_list_y.append(agumented_y)

        return augmented_list_x, augmented_list_y

    def perform_flips(self, images_x, images_y):

        if len(images_x) != len(images_y):
            raise ValueError("Wrong input. Image lists must be have the same length.")

        flipped_list_x = []
        flipped_list_y = []
        for image_x, image_y in zip(images_x, images_y):
            flipped_x, flipped_y = self.random_flip(image_x, image_y, 1)
            flipped_list_x.append(flipped_x)
            flipped_list_y.append(flipped_y)

        return flipped_list_x, flipped_list_y

    def perform_rotations(self, images_x, images_y):

        if len(images_x) != len(images_y):
            raise ValueError("Wrong input. Image lists must be have the same length.")

        rotated_list_x = []
        rotated_list_y = []
        for image_x, image_y in zip(images_x, images_y):
            angle = np.random.uniform(*self.theta_range)
            rotated_x = apply_affine_transform(image_x, theta=angle, fill_mode="constant", cval=0.)
            rotated_y = apply_affine_transform(image_y, theta=angle, fill_mode="constant", cval=0.)
            rotated_list_x.append(rotated_x)
            rotated_list_y.append(rotated_y)

        return rotated_list_x, rotated_list_y


    def perform_shifts(self, images_x, images_y):

        if len(images_x) != len(images_y):
            raise ValueError("Wrong input. Image lists must be have the same length.")

        shifted_list_x = []
        shifted_list_y = []
        for image_x, image_y in zip(images_x, images_y):
            width_shift = np.random.uniform(*self.x_shift_range)
            height_shift = np.random.uniform(*self.y_shift_range)
            rotated_x = apply_affine_transform(image_x, tx=width_shift, ty=height_shift, fill_mode="constant", cval=0.)
            rotated_y = apply_affine_transform(image_y, tx=width_shift, ty=height_shift, fill_mode="constant", cval=0.)
            shifted_list_x.append(rotated_x)
            shifted_list_y.append(rotated_y)

        return shifted_list_x, shifted_list_y


    def perform_shears(self, images_x, images_y):

        if len(images_x) != len(images_y):
            raise ValueError("Wrong input. Image lists must be have the same length.")

        sheared_list_x = []
        sheared_list_y = []
        for image_x, image_y in zip(images_x, images_y):
            shear = np.random.uniform(*self.shear_range)
            sheared_x = apply_affine_transform(image_x, shear=shear, fill_mode="constant", cval=0.)
            sheared_y = apply_affine_transform(image_y, shear=shear, fill_mode="constant", cval=0.)
            sheared_list_x.append(sheared_x)
            sheared_list_y.append(sheared_y)

        return sheared_list_x, sheared_list_y

    def apply_zoom(self, images_x, images_y):

        if len(images_x) != len(images_y):
            raise ValueError("Wrong input. Image lists must be have the same length.")

        zoomed_list_x = []
        zoomed_list_y = []
        for image_x, image_y in zip(images_x, images_y):
            zoom_x_value = np.random.uniform(*self.zx_range)
            zoom_y_value = np.random.uniform(*self.zy_range)
            zoomed_x = apply_affine_transform(image_x, zx=zoom_x_value, zy=zoom_y_value, fill_mode="constant", cval=0.)
            zoomed_y = apply_affine_transform(image_y, zx=zoom_x_value, zy=zoom_y_value, fill_mode="constant", cval=0.)
            zoomed_list_x.append(zoomed_x)
            zoomed_list_y.append(zoomed_y)

        return zoomed_list_x, zoomed_list_y

    def random_flip(self, x, y, axis):

        x = np.asarray(x).swapaxes(axis, 0)
        x = x[::-1, ...]
        x = x.swapaxes(0, axis)

        y = np.asarray(y).swapaxes(axis, 0)
        y = y[::-1, ...]
        y = y.swapaxes(0, axis)

        return x, y

    def visualize_data_augmentation(self, orig_x, orig_y, aug_x, aug_y):

        mult_aug_x, mult_aug_y = aug_x[len(orig_x):, ...], aug_y[len(orig_y):, ...]
        single_aug_x, single_aug_y = aug_x[:len(orig_x), ...], aug_y[:len(orig_y), ...]
        for x, y, aug1_x, aug1_y, aug2_x, aug2_y in zip(orig_x, orig_y, single_aug_x,
                                                        single_aug_y, mult_aug_x, mult_aug_y):
            row1 = np.concatenate([x[..., 0], x[..., 1], y[..., 0]], axis=1)
            row2 = np.concatenate([aug1_x[..., 0], aug1_x[..., 1], aug1_y[..., 0]], axis=1)
            row3 = np.concatenate([aug2_x[..., 0], aug2_x[..., 1], aug2_y[..., 0]], axis=1)
            full_image = np.concatenate([row1, row2, row3], axis=0)

            cv2.imshow("Image_comp", full_image)
            cv2.waitKey(0)