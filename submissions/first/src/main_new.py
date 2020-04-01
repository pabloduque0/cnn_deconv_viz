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
T1_COREG_MASK = "T1_bet_mask_rsfl.nii.gz"
T1_COREG_BRAIN = "T1_rsfl.nii.gz"

FLAIR_NAME = "FLAIR.nii.gz"
FLAIR_BET_NAME = "FLAIR_bet.nii.gz"
FLAIR_ENHANCED_NAME = "FLAIR-enhanced.nii.gz"

SLICER_PATH = "/slicer/" #old slicer
EDISON_PATH = "/add_ons/"
MODEL_FOLDER = "/wmhseg_example/models"


inputDir = '/input'
outputDir = '/output'

# Get masked images
base_command = "hd-bet -i {} -o {}"
process = subprocess.Popen(shlex.split(base_command.format(os.path.join(inputDir, 'pre', T1_NAME),
                                                           os.path.join(outputDir, T1_BET_NAME))),
                           stdout=subprocess.PIPE)
output, error = process.communicate()



# Co-register app
co_register_command = "./add_ons/coregister_app -3dspath {} -indir {} -images {} {} {} -outdir {}".format(SLICER_PATH,
                                                                                                          inputDir,
                                                                                                          FLAIR_NAME,
                                                                                                          T1_NAME, T1_MASK,
                                                                                                          outputDir)

subprocess.Popen(shlex.split(co_register_command), stdout=subprocess.PIPE)
output, error = process.communicate()

contrast_command = "./add_ons/edison_app -edisonpath {} -indir {} -images {} {} -outdir {}".format(EDISON_PATH,
                                                                                                   outputDir,
                                                                                                   FLAIR_NAME,
                                                                                                   T1_COREG_MASK,
                                                                                                   outputDir)

subprocess.Popen(shlex.split(contrast_command), stdout=subprocess.PIPE)
output, error = process.communicate()


# Load the image
flair = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(outputDir, FLAIR_ENHANCED_NAME)))
initial_size = flair.shape[1:]

REMOVE_TOP = int(np.ceil(flair.shape[0] * REMOVE_TOP_PRCTG))
REMOVE_BOT = int(np.ceil(flair.shape[0] * REMOVE_BOT_PRCTG))

flair = imageparser.preprocess_dataset_flair(flair, SLICE_SHAPE, flair.shape[0], REMOVE_TOP, REMOVE_BOT)
flair = np.expand_dims(flair, axis=-1)

common_mask = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(outputDir, T1_COREG_MASK)))
flair = flair * common_mask


t1 = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(outputDir, T1_COREG_BRAIN)))
t1 = imageparser.preprocess_dataset_t1(t1, SLICE_SHAPE, t1.shape[0], REMOVE_TOP, REMOVE_BOT)
t1 = np.expand_dims(t1, axis=-1)

data = np.concatenate([t1, flair], axis=-1)

enssemble_pred = None
for i in range(ENSSEMBLE_SIZE):
    model = load_model(os.path.join(MODEL_FOLDER, "model_20200326_longtrain_ksize11_ensemb_3_0_new_{}.hdf5".format(i)), custom_objects={"dice_coefficient_loss": metrics.dice_coefficient_loss,
                                                       "dice_coefficient": metrics.dice_coefficient,
                                                       "lession_recall": metrics.lession_recall})

    prediction = model.predict(data)
    bot_pad = np.zeros((REMOVE_BOT, *SLICE_SHAPE, 1))
    top_pad = np.zeros((REMOVE_TOP, *SLICE_SHAPE, 1))

    prediction = np.concatenate([bot_pad, prediction, top_pad])
    prediction = imageparser.resize_slices(prediction, initial_size)
    prediction = np.asanyarray(prediction)

    if not enssemble_pred:
        enssemble_pred = prediction

    else:
        enssemble_pred += prediction

enssemble_pred /= ENSSEMBLE_SIZE
enssemble_pred = np.round(enssemble_pred)

prediction = model.predict(data)
bot_pad = np.zeros((REMOVE_BOT, *SLICE_SHAPE, 1))
top_pad = np.zeros((REMOVE_TOP, *SLICE_SHAPE, 1))

prediction = np.concatenate([bot_pad, prediction, top_pad])
prediction = imageparser.resize_slices(prediction, initial_size)
prediction = np.asanyarray(prediction)

result_img = sitk.GetImageFromArray(np.squeeze(prediction))
sitk.WriteImage(result_img, os.path.join(outputDir, 'result.nii.gz'))



