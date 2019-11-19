import os
import gzip
import shutil
import cv2
import numpy as np
import subprocess
from scipy.stats import norm
import matplotlib.pyplot as plt
import SimpleITK



def get_all_sets_paths(dataset_paths):

    t1 = [row["t1_masked"] for row in dataset_paths]
    flair = [row["enhanced_masked"] for row in dataset_paths]
    labels = [row["label"] for row in dataset_paths]
    white_mask = [row["mask"] for row in dataset_paths]
    distance = [row["danielsson_dist"] for row in dataset_paths]

    return t1, flair, labels, white_mask, distance


def preprocess_dataset_t1(data_t1, slice_shape, n_slices, remove_top, remove_bot):

    resized_t1 = resize_slices(data_t1, slice_shape)
    resized_t1 = remove_top_bot_slices(resized_t1, n_slices,
                                            remove_n_top=remove_top,
                                            remove_n_bot=remove_bot)
    actual_n_slices = n_slices - remove_top - remove_bot
    stand_t1 = standarize(resized_t1, actual_n_slices)
    normalized_t1 = normalize_minmax(stand_t1, actual_n_slices)
    return normalized_t1


def preprocess_dataset_flair(data_flair, slice_shape, n_slices, remove_top, remove_bot):

    resized_flairs = resize_slices(data_flair, slice_shape)
    resized_flairs = remove_top_bot_slices(resized_flairs, n_slices,
                                                remove_n_top=remove_top,
                                                remove_n_bot=remove_bot)

    actual_n_slices = n_slices - remove_top - remove_bot
    stand_flairs = standarize(resized_flairs, actual_n_slices)
    norm_flairs = normalize_minmax(stand_flairs, actual_n_slices)

    return norm_flairs



def is_file_desired(file_name):
    possibilities = {"FLAIR_masked.nii.gz",
                        "FLAIR.nii.gz",
                        "FLAIR.nii",
                        "T1_masked.nii.gz",
                        "T1.nii.gz",
                        "T1.nii",
                        "distWMborder_Danielsson.nii.gz",
                        "distWMborder_Maurer.nii.gz",
                        "WMmask.nii.gz",
                        "FLAIR_enhanced_lb_masked.nii.gz",
                        "FLAIR_enhanced_lb.nii.gz"}
    return file_name in possibilities

def get_key(file_name):

    possibilities = {"FLAIR_masked.nii.gz": "flair_masked",
                     "FLAIR.nii.gz": "flair",
                     "T1_masked.nii.gz": "t1_masked",
                     "T1.nii.gz": "t1",
                     "distWMborder_Danielsson.nii.gz": "danielsson_dist",
                     "distWMborder_Maurer.nii.gz": "maurer_dist",
                     "WMmask.nii.gz": "mask",
                     "wmh.nii.gz": "label",
                     "FLAIR_enhanced_lb_masked.nii.gz": "enhanced_masked",
                     "FLAIR_enhanced_lb.nii.gz": "enhanced"}

    if file_name not in possibilities:
        return None

    return possibilities[file_name]

def get_all_images_itk(paths_list):
    images = []
    for path in paths_list:
        image = SimpleITK.imread(path)
        images.append(image)

    return images


def remove_top_bot_slices(dataset, n_slices, remove_n_top, remove_n_bot):

    if not isinstance(dataset, np.ndarray):
        dataset = np.asanyarray(dataset)

    output_images = None
    for idx in range(dataset.shape[0] // n_slices):
        if output_images is None:
            output_images = dataset[idx*n_slices + remove_n_bot : (idx+1)*n_slices - remove_n_top, ...]
        else:
            output_images = np.concatenate([output_images,
                                            dataset[idx * n_slices + remove_n_bot: (idx + 1) * n_slices - remove_n_top, ...]])

    output_images = np.asanyarray(output_images)
    return output_images


def process_all_images_np(paths_list, slice_shape, normalization=True):
    images = []
    for path in paths_list:
        image = SimpleITK.ReadImage(path)
        np_image = SimpleITK.GetArrayFromImage(image)
        np_image = np.swapaxes(np_image, 0, 2)
        resized = threed_resize(np_image, slice_shape)
        np_image = np.swapaxes(resized, 0, 2)

        if normalization:
            normalized = normalize_image(np_image)
            if normalized is not None:
                np_image = np.expand_dims(normalized, 4)
                images.append(np_image)
        else:
            np_image[np_image > 1.] = 0.0
            np_image = np.expand_dims(np_image, 4)
            images.append(np_image)

    return images


def resize_slices(slices_list, to_slice_shape):

    resized_slices = []

    for slice in slices_list:
        slice_copy = slice.copy()

        if slice.shape[0] < to_slice_shape[0]:
            diff = to_slice_shape[0] - slice.shape[0]
            if is_odd(diff):
                slice_copy = cv2.copyMakeBorder(slice_copy, diff//2, diff//2 + 1, 0, 0,
                                                cv2.BORDER_CONSTANT,
                                                value=0.0)
            else:
                slice_copy = cv2.copyMakeBorder(slice_copy, diff // 2, diff // 2, 0, 0,
                                                cv2.BORDER_CONSTANT,
                                                value=0.0)

        elif slice.shape[0] > to_slice_shape[0]:
            diff = slice.shape[0] - to_slice_shape[0]
            if is_odd(diff):
                slice_copy = slice_copy[diff//2 : -diff//2 + 1, :]
            else:
                slice_copy = slice_copy[diff // 2: -diff // 2, :]

        if slice.shape[1] < to_slice_shape[1]:
            diff = to_slice_shape[1] - slice.shape[1]
            if is_odd(diff):
                slice_copy = cv2.copyMakeBorder(slice_copy, 0, 0, diff // 2, diff // 2 + 1,
                                                cv2.BORDER_CONSTANT,
                                                value=0.0)
            else:
                slice_copy = cv2.copyMakeBorder(slice_copy, 0, 0, diff // 2, diff // 2,
                                                cv2.BORDER_CONSTANT,
                                                value=0.0)
        elif slice.shape[1] > to_slice_shape[1]:
            diff = slice.shape[1] - to_slice_shape[1]
            if is_odd(diff):
                slice_copy = slice_copy[:, diff // 2: -diff // 2 + 1]
            else:
                slice_copy = slice_copy[:, diff // 2: -diff // 2]

        resized_slices.append(slice_copy)

    return resized_slices

def is_odd(number):

    return number % 2 != 0

def threed_resize(image, slice_shape):

    all_slices = []
    for index in range(image.shape[2]):
        slice = image[:, :, index]
        resized = cv2.resize(slice, (slice_shape[1], slice_shape[0]), cv2.INTER_CUBIC)
        all_slices.append(resized)

    return np.asanyarray(all_slices)

def display_image(image):
    np_image = itk.GetArrayFromImage(image)

    np_image = np.swapaxes(np_image, 0, 2)
    np_image = np_image.astype(np.uint8)
    rows, columns, slices = np_image.shape
    for slice in range(slices):
        slice_image = np_image[:, :, slice]
        cv2.imshow('Image', slice_image)
        cv2.waitKey(0)

def normalize_minmax(images_list, slice_number):

    normalized_list = []

    np_list = np.asanyarray(images_list)
    for image_idx in range(np_list.shape[0]//slice_number):
        this_section = np_list[image_idx*slice_number:(image_idx+1)*slice_number, :, :]
        section_max = np.max(this_section)
        section_min = np.min(this_section)
        normalized = (this_section - section_min) / (section_max - section_min)
        normalized_list.append(normalized)

    normalized_list = np.concatenate(normalized_list)
    return normalized_list


def normalize_neg_pos_one(images_list, slice_number):

    normalized_list = []

    np_list = np.asanyarray(images_list)
    for image_idx in range(np_list.shape[0] // slice_number):
        this_section = np_list[image_idx * slice_number:(image_idx + 1) * slice_number, :, :]
        normalized = 2. * (this_section - np.min(this_section)) / np.ptp(this_section) - 1
        normalized_list.append(normalized)

    normalized_list = np.concatenate(normalized_list)
    return normalized_list



def normalize_quantile(flair_list, slice_number):

    normalized_images = []
    flair_list = np.asanyarray(flair_list)

    for image_idx in range(flair_list.shape[0] // slice_number):
        this_flair = flair_list[image_idx * slice_number:(image_idx + 1) * slice_number, :, :]

        flair_non_black = this_flair[this_flair > 0]
        lower_threshold, upper_threshold, upper_indexes, lower_indexes = get_thresholds_and_indexes(flair_non_black,
                                                                                                         99.7,
                                                                                                         this_flair)
        final_normalized = (this_flair - lower_threshold) / (upper_threshold - lower_threshold)
        final_normalized[upper_indexes] = 1.0
        final_normalized[lower_indexes] = 0.0
        print(np.max(final_normalized), np.min(final_normalized), np.mean(final_normalized))
        normalized_images.append(final_normalized)

    normalized_images = np.concatenate(normalized_images, axis=0)
    return normalized_images


def standarize(image_list, slice_number):

    standarized_imgs = []
    image_list = np.asanyarray(image_list)

    for image_idx in range(image_list.shape[0] // slice_number):
        this_image = image_list[image_idx * slice_number:(image_idx + 1) * slice_number, ...]
        stand_image = (this_image - np.mean(this_image)) / np.std(this_image)
        standarized_imgs.append(stand_image)

    standarized_imgs = np.concatenate(standarized_imgs, axis=0)
    return standarized_imgs

def study_intensity_values(flair_list, labels_list, slice_number):

    flair_list = np.asanyarray(flair_list)
    labels_list = np.asanyarray(labels_list)

    for image_idx in range(flair_list.shape[0] // slice_number):
        this_flair = flair_list[image_idx * slice_number:(image_idx + 1) * slice_number, :, :]
        this_labels = labels_list[image_idx * slice_number:(image_idx + 1) * slice_number, :, :]
        flair_labels_idx = np.where(this_labels[np.where(this_flair > 0)] > 0.)

        flair_perc = normalize_quantile(this_flair, this_labels, slice_number)
        flair_norm = np.asanyarray(normalize_minmax(this_flair, slice_number))
        flair_perc = np.ravel(flair_perc[flair_perc > 0])
        flair_norm = np.ravel(flair_norm[flair_norm > 0])

        fig, ax = plt.subplots(2, 1)
        ax[0].hist(flair_norm, bins=100, label="Norm_percentil", color="g", alpha=0.9)
        ax[0].set_title("Min-max Normalization")
        #ax_2y = ax[0].twinx()
        #ax_2y.hist(flair_perc[flair_labels_idx], bins=100, label="Norm_percentil", alpha=0.5, color="tab:red")
        #ax_2y.tick_params('y', colors='r')

        ax[1].hist(flair_perc, bins=100, label="Norm_minmax", color="g", alpha=0.9)
        ax[1].set_title("Thresholded Normalization")
        #ax2_2y = ax[1].twinx()
        #ax2_2y.hist(flair_norm[flair_labels_idx], bins=100, label="Norm_percentil", alpha=0.5, color="tab:red")
        #ax2_2y.tick_params('y', colors='r')
        plt.subplots_adjust(hspace=0.4)
        plt.show()

        """
        for flair, flair2 in zip(flair_perc, flair_norm):
            cv2.imshow("Flair-T1", np.concatenate([flair, flair2], axis=1))
            cv2.waitKey(0)
        """


def remove_third_label(labels_list):

    new_labels_list = []

    for image in labels_list:
        image[image > 1.] = 0.0
        new_labels_list.append(image)

    return new_labels_list



def generate_tophat(dataset):

    tophat_list = []
    kernel = np.ones((3, 3))
    for image in dataset:
        tophat = cv2.morphologyEx(image, cv2.MORPH_TOPHAT, kernel)
        tophat_list.append(np.expand_dims(tophat, axis=2))

    return tophat_list


def extract_all_brains(self):
    base_command = 'fsl5.0-bet '
    brain_str = 'brain_'
    for root, dirs, files in os.walk('../'):
        for file in files:
            filepath = root + '/' + file

            if '.nii' in file and file != 'wmh.nii' and 'mask' not in file:
                full_command = base_command + filepath + ' ' + root + '/' + brain_str + file
                process = subprocess.Popen(full_command.split(), stdout=subprocess.PIPE)
                output, error = process.communicate()

                print('OUTPUT: ', output)
                print('ERROR: ', error)


def extract_enhanced_brains(self):
    base_command = 'bet2 '
    brain_str = 'brain_'
    for root, dirs, files in os.walk('../'):
        for file in files:
            filepath = root + '/' + file

            if file == "FLAIR_enhanced_lb.nii.gz":
                full_command = base_command + filepath + ' ' + root + '/' + brain_str + file
                process = subprocess.Popen(full_command.split(), stdout=subprocess.PIPE)
                output, error = process.communicate()

                print('OUTPUT: ', output)
                print('ERROR: ', error)