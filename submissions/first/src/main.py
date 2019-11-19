import os
import SimpleITK as sitk
from keras.models import load_model
import numpy as np
import imageparser
import metrics

REMOVE_TOP = 4
REMOVE_BOT = 6
SLICE_SHAPE = (200, 200)

inputDir = '/input'
outputDir = '/output'

# Load the image
flair = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(inputDir, 'pre', 'FLAIR.nii.gz')))
initial_size = flair.shape[1:]
flair = imageparser.preprocess_dataset_flair(flair, SLICE_SHAPE, flair.shape[0], REMOVE_TOP, REMOVE_BOT)
flair = np.expand_dims(flair, axis=-1)
t1 = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(inputDir, 'pre', 'T1.nii.gz')))
t1 = imageparser.preprocess_dataset_t1(t1, SLICE_SHAPE, t1.shape[0], REMOVE_TOP, REMOVE_BOT)
t1 = np.expand_dims(t1, axis=-1)

data = np.concatenate([t1, flair], axis=-1)


model = load_model("/wmhseg_example/model_0.hdf5", custom_objects={"dice_coefficient_loss": metrics.dice_coefficient_loss,
                                                   "dice_coefficient": metrics.dice_coefficient,
                                                   "lession_recall": metrics.lession_recall})


prediction = model.predict(data)
bot_pad = np.zeros((REMOVE_BOT, *SLICE_SHAPE, 1))
top_pad = np.zeros((REMOVE_TOP, *SLICE_SHAPE, 1))

prediction = np.concatenate([bot_pad, prediction, top_pad])

diff_size = np.array(initial_size) - np.array()

if initial_size[0] > SLICE_SHAPE[0]:
    prediction = np.pad(prediction, (0, 0), (, ), (0, 0))
elif initial_size[0] < SLICE_SHAPE[0]:

if initial_size[1] > SLICE_SHAPE[1]:
    prediction = np.pad(prediction, (0, 0), (0, 0), (0, initial_size[1] - SLICE_SHAPE[1]))
elif initial_size[1] < SLICE_SHAPE[1]:
    prediction[:, : SLICE_SHAPE[1] - initial_size[1]]


np.pad(prediction, ((0, 0), (initial_size[0]-initial_size), ()))

result_img = sitk.GetImageFromArray(np.squeeze(prediction))
sitk.WriteImage(result_img, os.path.join(outputDir, 'result.nii.gz'))