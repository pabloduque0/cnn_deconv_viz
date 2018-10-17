from keras import models
from keras import layers
from contextlib import redirect_stdout
import os
from keras.callbacks import ModelCheckpoint, TensorBoard
from sklearn.model_selection import train_test_split
from keras.models import load_model
from keras.optimizers import Adam, SGD
from metrics import dice_coef, dice_coef_loss, weighted_crossentropy, predicted_count, ground_truth_count, ground_truth_sum, predicted_sum
import custom_layers


img_path = ""

inputs = layers.Input(shape=(10, 10, 2))
conv1 = layers.Conv2D(64, kernel_size=5, padding='same', kernel_initializer='he_normal', activation='relu')(inputs)
conv2 = layers.Conv2D(64, kernel_size=5, padding='same', kernel_initializer='he_normal', activation='relu')(conv1)
output, switches = custom_layers.MaxPoolingWithArgmax2D(pool_size=(2, 2))(conv2)
unpooling1 = custom_layers.unpooling2D(output, argmax=switches)

conv_trans1 = layers.Conv2DTranspose(64, kernel_size=5, padding='same', kernel_initializer='he_normal', activation='relu')(unpooling1)
conv_trans2 = layers.Conv2DTranspose(1, kernel_size=5, padding='same', kernel_initializer='he_normal', activation='relu')(conv_trans1)


model = models.Model(inputs=[inputs], outputs=[conv_trans2])

model.compile(optimizer=Adam(lr=0.000001), loss=dice_coef_loss, metrics=[dice_coef, weighted_crossentropy,
                                                                                   predicted_count, predicted_sum, ground_truth_count,
                                                                                 ground_truth_sum])