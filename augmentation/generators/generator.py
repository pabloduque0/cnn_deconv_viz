from keras import models
from keras import layers
from augmentation import metrics
from keras.optimizers import Adam
import keras.backend as K

def create_model(input_shape):

    input_layer = layers.Input(shape=input_shape)

    conv1 = layers.Conv2D(512, kernel_size=4, padding='same')(input_layer)
    conv1 = layers.LeakyReLU(alpha=0.2)(conv1)

    conv2 = layers.Conv2D(512, kernel_size=3, padding='same')(conv1)
    conv2 = layers.LeakyReLU(alpha=0.2)(conv2)

    block1_out = conv_up_block(conv2, 512)
    block1_out = layers.ZeroPadding2D(((1, 0), (1, 0)))(block1_out)
    block2_out = conv_up_block(block1_out, 512)
    block3_out = conv_up_block(block2_out, 512)
    block4_out = conv_up_block(block3_out, 256)
    block4_out = layers.ZeroPadding2D(((1, 0), (1, 0)))(block4_out)
    block5_out = conv_up_block(block4_out, 128)
    block6_out = conv_up_block(block5_out, 64)
    block7_out = conv_up_block(block6_out, 32)

    conv_last = layers.Conv2D(3, kernel_size=1, padding='same')(block7_out)
    model = models.Model(inputs=input_layer, outputs=conv_last)

    model.compile(Adam(lr=0.001, beta_1=0, beta_2=0.99), loss=metrics.wasserstein_loss)
    model.summary()

    return model


def conv_up_block(input_layer, filters):

    upsample = layers.UpSampling2D(size=(2, 2), interpolation="nearest")(input_layer)

    conv1 = layers.Conv2D(filters, kernel_size=3, padding='same')(upsample)
    conv1 = layers.LeakyReLU(alpha=0.2)(conv1)

    conv2 = layers.Conv2D(filters, kernel_size=3, padding='same')(conv1)
    conv2 = layers.LeakyReLU(alpha=0.2)(conv2)

    return conv2