from keras import models
from keras import layers
import os
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.models import load_model
from keras.optimizers import Adam
from networks.metrics import dice_coef, dice_coef_loss, weighted_crossentropy, predicted_count, \
    ground_truth_count, ground_truth_sum, predicted_sum, pixel_recall, custom_dice_coef, custom_dice_loss, obj_recall
from keras.losses import binary_crossentropy
import cv2
import numpy as np
from networks import custom_layers
import keras.backend as K
from networks.basenetwork import BaseNetwork

class EnssembleNets:

    def __init__(self, networks):
        self.networks = networks
        self.model = self.create_model()

    def create_model(self):

        outputs = [model.outputs[0] for model in self.networks]
        average = layers.Average()(outputs)
        model = models.Model(input_layer, average)


    def test(self):
        pass

