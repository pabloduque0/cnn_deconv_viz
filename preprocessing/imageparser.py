import os
import gzip
import shutil
import cv2
import numpy as np
import subprocess
from scipy.stats import norm
from constants import *
import matplotlib.pyplot as plt
import SimpleITK

class ImageParser():

    def __init__(self, path_utrech='../Utrecht/subjects',
                 path_singapore='../Singapore/subjects',
                 path_amsterdam='../GE3T/subjects'):
        self.path_utrech = path_utrech
        self.path_singapore = path_singapore
        self.path_amsterdam = path_amsterdam


    def get_all_image_paths(self):
        paths = []

        for root, dirs, files in os.walk('../'):
            for file in files:
                filepath = root + '/' + file

                if file.endswith('.gz') and file[:-3] not in files:
                    with gzip.open(filepath, 'rb') as f_in:
                        with open(filepath[:-3], 'wb') as f_out:
                            shutil.copyfileobj(f_in, f_out)

                if file.startswith('brain') and file.endswith('.nii'):
                    paths.append(filepath)

        return paths


    def get_all_images_and_labels(self):
        utrech_dataset = self.get_images_and_labels(self.path_utrech)
        singapore_dataset = self.get_images_and_labels(self.path_singapore)
        amsterdam_dataset = self.get_images_and_labels(self.path_amsterdam)

        return utrech_dataset, singapore_dataset, amsterdam_dataset


    def get_images_and_labels(self, path):
        full_dataset = []
        data_and_labels = {}

        for root, dirs, files in os.walk(path):
            for file in files:
                filepath = root + '/' + file
                key = self.get_key(file)
                if file == 'wmh.nii.gz':
                    data_and_labels[key] = filepath

                length = len(data_and_labels)
                if '/pre/' in filepath and self.is_file_desired(file) and length < 10 and length > 0:
                    data_and_labels[key] = filepath
                    if len(data_and_labels) == 10:
                        full_dataset.append(data_and_labels.copy())
                        print(data_and_labels)
                        data_and_labels.clear()

        return full_dataset

    def is_file_desired(self, file_name):
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

    def get_key(self, file_name):

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

    def get_all_images_itk(self, paths_list):
        images = []
        for path in paths_list:
            image = SimpleITK.imread(path)
            images.append(image)

        return images


    def remove_top_bot_slices(self, dataset, n_slices, remove_n_top=REMOVE_TOP, remove_n_bot=REMOVE_BOT):

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


    def process_all_images_np(self, paths_list, slice_shape, normalization=True):
        images = []
        for path in paths_list:
            image = SimpleITK.imread(path)
            np_image = SimpleITK.GetArrayFromImage(image)
            np_image = np.swapaxes(np_image, 0, 2)
            resized = self.threed_resize(np_image, slice_shape)
            np_image = np.swapaxes(resized, 0, 2)

            if normalization:
                normalized = self.normalize_image(np_image)
                if normalized is not None:
                    np_image = np.expand_dims(normalized, 4)
                    images.append(np_image)
            else:
                np_image[np_image > 1.] = 0.0
                np_image = np.expand_dims(np_image, 4)
                images.append(np_image)

        return images

    def get_all_images_np_twod(self, paths_list):

        slices_list = []
        for path in paths_list:
            image = SimpleITK.ReadImage(path)
            np_image = SimpleITK.GetArrayFromImage(image)
            if np_image.shape[1:] == (232, 256):
                np_image = np.swapaxes(np_image, 1, 2)
                print('Corrected axises')

            for slice in np_image:
                slices_list.append(slice)

        return slices_list


    def resize_slices(self, slices_list, to_slice_shape):

        resized_slices = []

        for slice in slices_list:
            slice_copy = slice.copy()

            if slice.shape[0] < to_slice_shape[0]:
                diff = to_slice_shape[0] - slice.shape[0]
                if self.is_odd(diff):
                    slice_copy = cv2.copyMakeBorder(slice_copy, diff//2, diff//2 + 1, 0, 0,
                                                    cv2.BORDER_CONSTANT,
                                                    value=0.0)
                else:
                    slice_copy = cv2.copyMakeBorder(slice_copy, diff // 2, diff // 2, 0, 0,
                                                    cv2.BORDER_CONSTANT,
                                                    value=0.0)

            elif slice.shape[0] > to_slice_shape[0]:
                diff = slice.shape[0] - to_slice_shape[0]
                if self.is_odd(diff):
                    slice_copy = slice_copy[diff//2 : -diff//2 + 1, :]
                else:
                    slice_copy = slice_copy[diff // 2: -diff // 2, :]

            if slice.shape[1] < to_slice_shape[1]:
                diff = to_slice_shape[1] - slice.shape[1]
                if self.is_odd(diff):
                    slice_copy = cv2.copyMakeBorder(slice_copy, 0, 0, diff // 2, diff // 2 + 1,
                                                    cv2.BORDER_CONSTANT,
                                                    value=0.0)
                else:
                    slice_copy = cv2.copyMakeBorder(slice_copy, 0, 0, diff // 2, diff // 2,
                                                    cv2.BORDER_CONSTANT,
                                                    value=0.0)
            elif slice.shape[1] > to_slice_shape[1]:
                diff = slice.shape[1] - to_slice_shape[1]
                if self.is_odd(diff):
                    slice_copy = slice_copy[:, diff // 2: -diff // 2 + 1]
                else:
                    slice_copy = slice_copy[:, diff // 2: -diff // 2]

            resized_slices.append(slice_copy)

        return resized_slices

    def is_odd(self, number):

        return number % 2 != 0

    def threed_resize(self, image, slice_shape):

        all_slices = []
        for index in range(image.shape[2]):
            slice = image[:, :, index]
            resized = cv2.resize(slice, (slice_shape[1], slice_shape[0]), cv2.INTER_CUBIC)
            all_slices.append(resized)

        return np.asanyarray(all_slices)

    def display_image(self, image):
        np_image = itk.GetArrayFromImage(image)

        np_image = np.swapaxes(np_image, 0, 2)
        np_image = np_image.astype(np.uint8)
        rows, columns, slices = np_image.shape
        for slice in range(slices):
            slice_image = np_image[:, :, slice]
            cv2.imshow('Image', slice_image)
            cv2.waitKey(0)

    def normalize_minmax(self, images_list, slice_number):

        normalized_list = []

        np_list = np.asanyarray(images_list)
        for image_idx in range(np_list.shape[0]//slice_number):
            this_section = np_list[image_idx*slice_number:(image_idx+1)*slice_number, :, :]
            flattened = np.ravel(this_section)
            non_black = flattened[flattened > 0]
            upper_threshold = np.max(non_black)
            normalized = (this_section) / (upper_threshold)
            normalized_list.append(normalized)

        normalized_list = np.concatenate(normalized_list)
        return normalized_list

    def normalize_quantile(self, flair_list, slice_number):

        normalized_images = []
        flair_list = np.asanyarray(flair_list)

        for image_idx in range(flair_list.shape[0] // slice_number):
            this_flair = flair_list[image_idx * slice_number:(image_idx + 1) * slice_number, :, :]

            flair_non_black = this_flair[this_flair > 0]
            lower_threshold, upper_threshold, upper_indexes, lower_indexes = self.get_thresholds_and_indexes(flair_non_black,
                                                                                                             99.7,
                                                                                                             this_flair)
            final_normalized = (this_flair - lower_threshold) / (upper_threshold - lower_threshold)
            final_normalized[upper_indexes] = 1.0
            final_normalized[lower_indexes] = 0.0
            print(np.max(final_normalized), np.min(final_normalized), np.mean(final_normalized))
            normalized_images.append(final_normalized)

        normalized_images = np.concatenate(normalized_images, axis=0)
        return normalized_images


    """
    def find_best_quantile_normalization(self, image, non_black, labels_idx):

        flair_not_labels_idx = np.where(np.delete(non_black, labels_idx) != None)

        upper_base = 95.
        max_difference = 0
        best_upper_base = None

        for i in range((int(100 - upper_base) * 10)):
            lower_threshold, upper_threshold, upper_indexes, lower_indexes = self.get_thresholds_and_indexes(non_black,
                                                                                                        upper_base)
            normalized_perc = (non_black - lower_threshold) / (upper_threshold - lower_threshold)
            normalized_perc[upper_indexes] = 1.0
            normalized_perc[lower_indexes] = 0.0

            normalized_minmax = (non_black - np.min(non_black)) / (np.max(non_black) - np.min(non_black))

            mean_perc, std = norm.fit(normalized_perc[flair_not_labels_idx])
            mean_idxs, std = norm.fit(normalized_perc[labels_idx])
            diff_perc = abs(mean_perc - mean_idxs)

            mean_minmax, std = norm.fit(normalized_minmax[flair_not_labels_idx])
            mean_minmax_idxs, std = norm.fit(normalized_minmax[labels_idx])
            diff_minmax = abs(mean_minmax - mean_minmax_idxs)

            if abs(diff_perc - diff_minmax) > max_difference:
                max_difference = abs(diff_perc - diff_minmax)
                best_upper_base = upper_base

            upper_base += .1


        print("best_upper_base: ", best_upper_base)

        return final_normalized
    """

    def get_thresholds_and_indexes(self, non_black, upper_perc, full_image=None, lower_perc=0.5):

        lower_threshold = np.percentile(non_black, lower_perc)
        upper_threshold = np.percentile(non_black, upper_perc)

        if full_image is None:
            upper_indexes = np.where(non_black >= upper_threshold)
            lower_indexes = np.where(non_black <= lower_threshold)
        else:
            upper_indexes = np.where(full_image >= upper_threshold)
            lower_indexes = np.where(full_image <= lower_threshold)

        return lower_threshold, upper_threshold, upper_indexes, lower_indexes


    def standarize(self, image_list, slice_number):

        standarized_imgs = []
        image_list = np.asanyarray(image_list)

        for image_idx in range(image_list.shape[0] // slice_number):
            this_image = image_list[image_idx * slice_number:(image_idx + 1) * slice_number, ...]
            stand_image = (this_image - np.mean(this_image)) / np.std(this_image)
            standarized_imgs.append(stand_image)

        standarized_imgs = np.concatenate(standarized_imgs, axis=0)
        return standarized_imgs

    def study_intensity_values(self, flair_list, labels_list, slice_number):

        flair_list = np.asanyarray(flair_list)
        labels_list = np.asanyarray(labels_list)

        for image_idx in range(flair_list.shape[0] // slice_number):
            this_flair = flair_list[image_idx * slice_number:(image_idx + 1) * slice_number, :, :]
            this_labels = labels_list[image_idx * slice_number:(image_idx + 1) * slice_number, :, :]
            flair_labels_idx = np.where(this_labels[np.where(this_flair > 0)] > 0.)

            flair_perc = self.normalize_quantile(this_flair, this_labels, slice_number)
            flair_norm = np.asanyarray(self.normalize_minmax(this_flair, slice_number))
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


    def remove_third_label(self, labels_list):

        new_labels_list = []

        for image in labels_list:
            image[image > 1.] = 0.0
            new_labels_list.append(image)

        return new_labels_list



    def generate_tophat(self, dataset):

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