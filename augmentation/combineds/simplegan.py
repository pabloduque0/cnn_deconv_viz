import numpy as np
from augmentation.discriminators import discriminator
from augmentation.generators import generator
from keras import layers
from keras import models
from keras import optimizers
from augmentation import metrics
import cv2
import os
from keras import losses
from keras import optimizers
import tensorflow as tf
from augmentation import utils
from augmentation.mothergan import MotherGAN

class SimpleGAN(MotherGAN):

    def __init__(self, img_shape, noise_shape):

        discriminator_model = discriminator.create_model(img_shape)
        # Build and compile the discriminator
        discriminator_model.compile(optimizers.Adam(lr=0.00001, beta_1=0, beta_2=0.99),
                                   loss=losses.binary_crossentropy, metrics=['accuracy'])

        # Build and compile the generator
        generator_model = generator.create_model(noise_shape)

        # The generator takes noise as input and generated imgs
        z = layers.Input(shape=noise_shape)
        image = generator_model(z)

        # For the combined model we will only train the generator
        discriminator_model.trainable = False
        # The valid takes generated images as input and determines validity
        valid = discriminator_model(image)

        # The combined model  (stacked generator and discriminator) takes
        # noise as input => generates images => determines validity
        combined_model = models.Model(z, valid)
        combined_model.compile(optimizers.Adam(lr=0.00001, beta_1=0, beta_2=0.99), loss=metrics.generator_loss)

        super().__init__(generator_model, discriminator_model, combined_model, img_shape, noise_shape)
