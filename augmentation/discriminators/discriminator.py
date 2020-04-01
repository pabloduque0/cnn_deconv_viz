from keras import models
from keras import layers
import tensorflow as tf
from keras.optimizers import Adam
from augmentation import metrics
from keras import losses


def create_model(input_shape):

    input_layer = layers.Input(shape=input_shape)

    conv1 = layers.Conv2D(16, kernel_size=1, padding='same', activation='relu')(input_layer)
    conv1 = layers.LeakyReLU(alpha=0.2)(conv1)

    block1_out = create_conv_block(conv1, (32, 64))
    block2_out = create_conv_block(block1_out, (64, 128))
    block3_out = create_conv_block(block2_out, (128, 256))
    block4_out = create_conv_block(block3_out, (256, 512))
    block5_out = create_conv_block(block4_out, (512, 512))

    normalized = layers.Lambda(lambda x: MiniBatchStddev(x))(block5_out)
    conv2 = layers.Conv2D(512, kernel_size=3, padding='same')(normalized)
    conv2 = layers.LeakyReLU(alpha=0.2)(conv2)

    conv3 = layers.Conv2D(512, kernel_size=4, padding='same')(conv2)
    conv3 = layers.LeakyReLU(alpha=0.2)(conv3)

    flatten = layers.Flatten()(conv3)
    dense = layers.Dense(1)(flatten)
    final_activation = layers.Activation("sigmoid")(dense)

    model = models.Model(inputs=input_layer, outputs=final_activation)

    model.summary()

    return model



def create_conv_block(input_layer, filters):

    conv1 = layers.Conv2D(filters[0], kernel_size=3, padding='same')(input_layer)
    conv1 = layers.LeakyReLU(alpha=0.2)(conv1)
    conv2 = layers.Conv2D(filters[1], kernel_size=3, padding='same')(conv1)
    conv2 = layers.LeakyReLU(alpha=0.2)(conv2)

    output_layer = layers.AveragePooling2D(pool_size=(2, 2))(conv2)

    return output_layer

def MiniBatchStddev(x, group_size=5): #again position of channels matter!
    group_size = tf.minimum(group_size, tf.shape(input=x)[0])# Minibatch must be divisible by (or smaller than) group_size.
    s = x.shape                                             # [NCHW]  Input shape.
    y = tf.reshape(x, [group_size, -1, s[1], s[2], s[3]])   # [GMCHW] Split minibatch into M groups of size G.
    y = tf.cast(y, tf.float32)                              # [GMCHW] Cast to FP32.
    y -= tf.reduce_mean(input_tensor=y, axis=0, keepdims=True)           # [GMCHW] Subtract mean over group.
    y = tf.reduce_mean(input_tensor=tf.square(y), axis=0)                # [MCHW]  Calc variance over group.
    y = tf.sqrt(y + 1e-8)                                   # [MCHW]  Calc stddev over group.
    y = tf.reduce_mean(input_tensor=y, axis=[1,2,3], keepdims=True)      # [M111]  Take average over fmaps and pixels.
    y = tf.cast(y, x.dtype)                                 # [M111]  Cast back to original data type.
    y = tf.tile(y, [group_size, s[1], s[2], 1])             # [N1HW]  Replicate over group and pixels.
    return tf.concat([x, y], axis=-1)

