import os
import SimpleITK as sitk
import numpy as np
import subprocess
import cv2
import shlex


T1_NAME = "T1.nii.gz"
T1_BET_NAME = "T1_bet.nii.gz"
T1_MASK = "T1_bet_mask.nii.gz"
T1_COREG_MASK = "T1_bet_mask_rsfl.nii.gz"
T1_COREG_BRAIN = "T1_rsfl.nii.gz"

FLAIR_NAME = "FLAIR.nii.gz"
FLAIR_BET_NAME = "FLAIR_bet.nii.gz"
FLAIR_ENHANCED_NAME = "FLAIR-enhanced.nii.gz"

base_path = os.path.abspath("../../new_data_set/")
DATASETS = ["Utrecht/", "Singapore/", "GE3T/"]
RUNNER_PATH = os.path.join(base_path, "contrast_enhancement-pablo/ejecutables")
SLICER_PATH = os.path.join(base_path, "slicer")


for dataset in DATASETS:
    this_path = os.path.join(base_path, dataset)
    for dirpath, dirs, files in os.walk(this_path):
        if T1_NAME in files and FLAIR_NAME in files:
            print(dirpath.split("/")[4:7])
            # Get masked images
            print("Running HD-BET")
            base_command = "hd-bet -i {}"
            process1 = subprocess.Popen(shlex.split(base_command.format(os.path.join(dirpath, T1_NAME), dirpath)),
                                       stdout=subprocess.PIPE)
            output1, error1 = process1.communicate()
            print(str(output1), str(error1))


            print("Running co-register")
            # Co-register app
            co_register_command = "{} -3dspath {} -indir {} -images {} {} {} -outdir {}".format(
                os.path.join(RUNNER_PATH, "coregister_app"),
                SLICER_PATH,
                dirpath,
                FLAIR_NAME,
                T1_NAME, T1_MASK,
                dirpath)

            process2 = subprocess.Popen(shlex.split(co_register_command), stdout=subprocess.PIPE)
            output2, error2 = process2.communicate()
            print(str(output2), str(error2))

            print("Running contrast enhancement")
            contrast_command = "{} -edisonpath {} -indir {} -images {} {} -outdir {}".format(
                os.path.join(RUNNER_PATH, "edison_app"),
                RUNNER_PATH,
                dirpath,
                FLAIR_NAME,
                T1_COREG_MASK,
                dirpath)

            process3 = subprocess.Popen(shlex.split(contrast_command), stdout=subprocess.PIPE)
            output3, error3 = process3.communicate()
            print(str(output3), str(error3))

