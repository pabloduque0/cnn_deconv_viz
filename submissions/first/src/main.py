import os
import SimpleITK as sitk
from keras.models import load_model
import numpy as np
import imageparser
import metrics
import subprocess
import cv2
import shlex

REMOVE_TOP_PRCTG = 0.09
REMOVE_BOT_PRCTG = 0.14
SLICE_SHAPE = (200, 200)

ENSSEMBLE_SIZE = 3

T1_NAME = "T1.nii.gz"
T1_BET_NAME = "T1_bet.nii.gz"
T1_MASK = "T1_bet_mask.nii.gz"
FLAIR_NAME = "FLAIR.nii.gz"
FLAIR_BET_NAME = "FLAIR_bet.nii.gz"

MODEL_FOLDER = "/wmhseg_example/models"

inputDir = '/input'
outputDir = '/output'

# Get masked images
base_command = "hd-bet -i {} -o {} -device cpu -mode fast -tta 0"
process = subprocess.Popen(shlex.split(base_command.format(os.path.join(inputDir, 'pre', T1_NAME),
                                                           os.path.join(outputDir, T1_BET_NAME))),
                           stdout=subprocess.PIPE)
output, error = process.communicate()


process = subprocess.Popen(shlex.split(base_command.format(os.path.join(inputDir, 'pre', T1_NAME),
                                                           os.path.join(outputDir, T1_BET_NAME))),
                           stdout=subprocess.PIPE)
output, error = process.communicate()



# Load the image
flair = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(outputDir, FLAIR_BET_NAME)))
initial_size = flair.shape[1:]

REMOVE_TOP = int(np.ceil(flair.shape[0] * REMOVE_TOP_PRCTG))
REMOVE_BOT = int(np.ceil(flair.shape[0] * REMOVE_BOT_PRCTG))

flair = imageparser.preprocess_dataset_flair(flair, SLICE_SHAPE, flair.shape[0], REMOVE_TOP, REMOVE_BOT)
flair = np.expand_dims(flair, axis=-1)
t1 = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(outputDir, T1_BET_NAME)))
t1 = imageparser.preprocess_dataset_t1(t1, SLICE_SHAPE, t1.shape[0], REMOVE_TOP, REMOVE_BOT)
t1 = np.expand_dims(t1, axis=-1)

data = np.concatenate([t1, flair], axis=-1)

model = load_model(os.path.join(MODEL_FOLDER, "model_20200326_longtrain_ksize11_ensemb_3_0_new_0.hdf5"),
                   custom_objects={"dice_coefficient_loss": metrics.dice_coefficient_loss,
                                   "dice_coefficient": metrics.dice_coefficient,
                                   "lession_recall": metrics.lession_recall})

prediction = model.predict(data)

result_img = sitk.GetImageFromArray(np.squeeze(prediction))
sitk.WriteImage(result_img, os.path.join(outputDir, 'result.nii.gz'))



