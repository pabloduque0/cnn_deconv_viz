import numpy as np
import cv2
from keras.models import Model, Sequential
from keras.layers import Input, Dense, Reshape, Flatten
from keras.layers.convolutional import Convolution2D, Conv2DTranspose
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras import layers
from keras.optimizers import Adam
from keras.datasets import mnist
from keras import backend as K
import matplotlib.pyplot as plt
from time import time
import tensorflow as tf
import os
from keras.callbacks import TensorBoard
from datetime import datetime
from keras.layers.merge import _Merge
from keras import backend as K
from functools import partial


EPOCHS = 20000
BATCH_SIZE = 8
# The training ratio is the number of discriminator updates
# per generator update. The paper uses 5.
TRAINING_RATIO = 8
GRADIENT_PENALTY_WEIGHT = 10  # As per the paper
INPUT_LEN = 128


def make_discriminator():
    """Creates a discriminator model that takes an image as input and outputs a single
    value, representing whether the input is real or generated. Unlike normal GANs, the
    output is not sigmoid and does not represent a probability! Instead, the output
    should be as large and negative as possible for generated inputs and as large and
    positive as possible for real inputs.
    Note that the improved WGAN paper suggests that BatchNormalization should not be
    used in the discriminator."""

    model = Sequential()
    model.add(Convolution2D(32, 5, padding='same', strides=[2, 2], input_shape=(256, 256, 2)))
    model.add(LeakyReLU())

    model.add(Convolution2D(64, 5, kernel_initializer='he_normal', strides=[2, 2], padding='same'))
    model.add(LeakyReLU())

    model.add(Convolution2D(128, 5, kernel_initializer='he_normal', padding='same', strides=[2, 2]))
    model.add(LeakyReLU())

    model.add(Convolution2D(256, 5, kernel_initializer='he_normal', padding='same', strides=[2, 2]))
    model.add(LeakyReLU())

    model.add(Convolution2D(512, 5, kernel_initializer='he_normal', padding='same', strides=[2, 2]))
    model.add(LeakyReLU())

    model.add(Convolution2D(1024, 5, kernel_initializer='he_normal', padding='same', strides=[2, 2]))
    model.add(LeakyReLU())

    model.add(Flatten())
    # model.add(Dense(1024 * 4 * 4, kernel_initializer='he_normal'))
    # model.add(LeakyReLU())
    model.add(Dense(1, kernel_initializer='he_normal'))

    return model


def make_generator():
    """Creates a generator model that takes a 128-dimensional noise vector as a "seed",
    and outputs images of size 256x256x3."""
    model = Sequential()

    model.add(Dense(6* 6* 2048, input_dim=INPUT_LEN))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    model.add(Reshape((6, 6, 2048), input_shape=(6* 6* 2048,)))
    bn_axis = -1

    model.add(Conv2DTranspose(1024, 5, strides=2, padding='same'))
    model.add(BatchNormalization(axis=bn_axis))
    model.add(LeakyReLU())

    model.add(Conv2DTranspose(512, 5, strides=2, padding='same'))
    model.add(BatchNormalization(axis=bn_axis))
    model.add(LeakyReLU())

    model.add(layers.ZeroPadding2D(padding=((1, 0), (1, 0))))

    model.add(Conv2DTranspose(256, 5, strides=2, padding='same'))
    model.add(BatchNormalization(axis=bn_axis))
    model.add(LeakyReLU())

    model.add(Conv2DTranspose(128, 5, strides=2, padding='same'))
    model.add(BatchNormalization(axis=bn_axis))
    model.add(LeakyReLU())

    model.add(Conv2DTranspose(2, 5, strides=2, padding='same', activation='tanh'))
    # El output de esta última es 256x256x3

    return model


def save_imgs(generator, discriminator, imgs_path, epoch, noise_shape, total_images=100, get_n_best=10):

    noise = np.random.normal(0, 1, (total_images, *noise_shape))
    gen_imgs = generator.predict(noise)
    images_mark = discriminator.predict(gen_imgs).reshape((total_images))
    order = np.argsort(-images_mark)[:get_n_best]
    images_final = gen_imgs[order, ...]

    for i in range(get_n_best):
        img_name = "%d_%d_generated_img.png" % (epoch, i)
        this_img = images_final[i, ...]
        re_scaled = (this_img - np.min(this_img)) * 255 / (np.max(this_img) - np.min(this_img))
        cv2.imwrite(os.path.join(imgs_path, img_name),
                    np.concatenate([re_scaled[:, :, 0], re_scaled[:, :, 1]], axis=1))



def wasserstein_loss(y_true, y_pred):
    """Calculates the Wasserstein loss for a sample batch.
    The Wasserstein loss function is very simple to calculate. In a standard GAN, the
    discriminator has a sigmoid output, representing the probability that samples are
    real or generated. In Wasserstein GANs, however, the output is linear with no
    activation function! Instead of being constrained to [0, 1], the discriminator wants
    to make the distance between its output for real and generated samples as
    large as possible.
    The most natural way to achieve this is to label generated samples -1 and real
    samples 1, instead of the 0 and 1 used in normal GANs, so that multiplying the
    outputs by the labels will give you the loss immediately.
    Note that the nature of this loss means that it can be (and frequently will be)
    less than 0."""
    return K.mean(y_true * y_pred)


def gradient_penalty_loss(y_true, y_pred, averaged_samples, gradient_penalty_weight):
    """Calculates the gradient penalty loss for a batch of "averaged" samples.
    In Improved WGANs, the 1-Lipschitz constraint is enforced by adding a term to the
    loss function that penalizes the network if the gradient norm moves away from 1.
    However, it is impossible to evaluate this function at all points in the input
    space. The compromise used in the paper is to choose random points on the lines
    between real and generated samples, and check the gradients at these points. Note
    that it is the gradient w.r.t. the input averaged samples, not the weights of the
    discriminator, that we're penalizing!
    In order to evaluate the gradients, we must first run samples through the generator
    and evaluate the loss. Then we get the gradients of the discriminator w.r.t. the
    input averaged samples. The l2 norm and penalty can then be calculated for this
    gradient.
    Note that this loss function requires the original averaged samples as input, but
    Keras only supports passing y_true and y_pred to loss functions. To get around this,
    we make a partial() of the function with the averaged_samples argument, and use that
    for model training."""
    # first get the gradients:
    #   assuming: - that y_pred has dimensions (batch_size, 1)
    #             - averaged_samples has dimensions (batch_size, nbr_features)
    # gradients afterwards has dimension (batch_size, nbr_features), basically
    # a list of nbr_features-dimensional gradient vectors
    gradients = K.gradients(y_pred, averaged_samples)[0]
    # compute the euclidean norm by squaring ...
    gradients_sqr = K.square(gradients)
    #   ... summing over the rows ...
    gradients_sqr_sum = K.sum(gradients_sqr, axis=np.arange(1, len(gradients_sqr.shape)))
    #   ... and sqrt
    gradient_l2_norm = K.sqrt(gradients_sqr_sum)
    # compute lambda * (1 - ||grad||)^2 still for each single sample
    gradient_penalty = gradient_penalty_weight * K.square(1 - gradient_l2_norm)
    # return the mean as loss over all the batch samples
    return K.mean(gradient_penalty)

def tile_images(image_stack):
    """Given a stacked tensor of images, reshapes them into a horizontal tiling for
    display."""
    assert len(image_stack.shape) == 3
    image_list = [image_stack[i, :, :] for i in range(image_stack.shape[0])]
    tiled_images = np.concatenate(image_list, axis=1)
    return tiled_images


class RandomWeightedAverage(_Merge):
    """Takes a randomly-weighted average of two tensors. In geometric terms, this
    outputs a random point on the line between each pair of input points.
    Inheriting from _Merge is a little messy but it was the quickest solution I could
    think o Improvements appreciated."""

    def _merge_function(self, inputs):
        weights = K.random_uniform((BATCH_SIZE, 1, 1, 1))
        return (weights * inputs[0]) + ((1 - weights) * inputs[1])



from augmentation.combineds import wassersteingan
from augmentation.combineds import wg_gp_gan
import numpy as np
from preprocessing.imageparser import ImageParser
from constants import *
import gc
import os
import cv2

parser = ImageParser(path_utrech='../../Utrecht/subjects',
                     path_singapore='../../Singapore/subjects',
                     path_amsterdam='../../GE3T/subjects')
utrech_dataset, singapore_dataset, amsterdam_dataset = parser.get_all_images_and_labels()

t1_utrecht, flair_utrecht, labels_utrecht, white_mask_utrecht, distance_utrecht = parser.get_all_sets_paths(utrech_dataset)
t1_singapore, flair_singapore, labels_singapore, white_mask_singapore, distance_singapore = parser.get_all_sets_paths(singapore_dataset)
t1_amsterdam, flair_amsterdam, labels_amsterdam, white_mask_amsterdam, distance_amsterdam = parser.get_all_sets_paths(amsterdam_dataset)

slice_shape = SLICE_SHAPE

print('Utrecht: ', len(t1_utrecht), len(flair_utrecht), len(labels_utrecht))
print('Singapore: ', len(t1_singapore), len(flair_singapore), len(labels_singapore))
print('Amsterdam: ', len(t1_amsterdam), len(flair_amsterdam), len(labels_amsterdam))



rm_extra_top = 14
rm_extra_bot = 17

rm_extra_amsterdam_bot = 21
rm_extra_amsterdam_top = 14
final_label_imgs = parser.preprocess_all_labels([labels_utrecht,
                                                 labels_singapore,
                                                 labels_amsterdam], slice_shape, [UTRECH_N_SLICES,
                                                                                  SINGAPORE_N_SLICES,
                                                                                  AMSTERDAM_N_SLICES],
                                                REMOVE_TOP + rm_extra_top,
                                                REMOVE_BOT + rm_extra_bot,
                                                (rm_extra_amsterdam_top, rm_extra_amsterdam_bot))

'''

T1 DATA

'''
rm_total = (REMOVE_TOP + REMOVE_BOT) + rm_extra_top + rm_extra_bot
utrecht_normalized_t1 = parser.preprocess_dataset_t1(t1_utrecht, slice_shape, UTRECH_N_SLICES,
                                                     REMOVE_TOP + rm_extra_top, REMOVE_BOT + rm_extra_bot, norm_type="stand")
utrecht_normalized_t1 = parser.normalize_neg_pos_one(utrecht_normalized_t1, UTRECH_N_SLICES - rm_total)

singapore_normalized_t1 = parser.preprocess_dataset_t1(t1_singapore, slice_shape, SINGAPORE_N_SLICES,
                                                       REMOVE_TOP + rm_extra_top, REMOVE_BOT + rm_extra_bot, norm_type="stand")
singapore_normalized_t1 = parser.normalize_neg_pos_one(singapore_normalized_t1, SINGAPORE_N_SLICES - rm_total)

amsterdam_normalized_t1 = parser.preprocess_dataset_t1(t1_amsterdam, slice_shape, AMSTERDAM_N_SLICES,
                                                       REMOVE_TOP + rm_extra_top + rm_extra_amsterdam_top,
                                                       REMOVE_BOT + rm_extra_bot + rm_extra_amsterdam_bot, norm_type="stand")
amsterdam_normalized_t1 = parser.normalize_neg_pos_one(amsterdam_normalized_t1,
                                                       AMSTERDAM_N_SLICES - rm_total - rm_extra_amsterdam_bot - rm_extra_amsterdam_top)
del t1_utrecht, t1_singapore, t1_amsterdam

'''

FLAIR DATA

'''


utrecht_stand_flairs = parser.preprocess_dataset_flair(flair_utrecht, slice_shape, UTRECH_N_SLICES,
                                                       REMOVE_TOP + rm_extra_top, REMOVE_BOT + rm_extra_bot, norm_type="stand")
utrecht_stand_flairs = parser.normalize_neg_pos_one(utrecht_stand_flairs, UTRECH_N_SLICES - rm_total)

singapore_stand_flairs = parser.preprocess_dataset_flair(flair_singapore, slice_shape, SINGAPORE_N_SLICES,
                                                       REMOVE_TOP + rm_extra_top, REMOVE_BOT + rm_extra_bot, norm_type="stand")
singapore_stand_flairs = parser.normalize_neg_pos_one(singapore_stand_flairs, SINGAPORE_N_SLICES - rm_total)

amsterdam_stand_flairs = parser.preprocess_dataset_flair(flair_amsterdam, slice_shape, AMSTERDAM_N_SLICES,
                                                         REMOVE_TOP + rm_extra_top + rm_extra_amsterdam_top,
                                                         REMOVE_BOT + rm_extra_bot + rm_extra_amsterdam_bot, norm_type="stand")
amsterdam_stand_flairs = parser.normalize_neg_pos_one(amsterdam_stand_flairs,
                                                      AMSTERDAM_N_SLICES - rm_total - rm_extra_amsterdam_bot - rm_extra_amsterdam_top)

del flair_utrecht, flair_singapore, flair_amsterdam


'''

DATA CONCAT

'''
normalized_t1 = np.concatenate([utrecht_normalized_t1,
                                singapore_normalized_t1,
                                amsterdam_normalized_t1], axis=0)

normalized_flairs = np.concatenate([utrecht_stand_flairs,
                                    singapore_stand_flairs,
                                    amsterdam_stand_flairs], axis=0)

del utrecht_normalized_t1, singapore_normalized_t1, amsterdam_normalized_t1
del utrecht_stand_flairs, singapore_stand_flairs, amsterdam_stand_flairs

data_t1 = np.expand_dims(np.asanyarray(normalized_t1), axis=3)
data_flair = np.expand_dims(np.asanyarray(normalized_flairs), axis=3)
all_data = np.concatenate([data_t1, data_flair], axis=3)


images = all_data
n_images = len(images)


generator = make_generator()
discriminator = make_discriminator()

for layer in discriminator.layers:
    layer.trainable = False

discriminator.trainable = False
generator_input = Input(shape=(INPUT_LEN,))
generator_layers = generator(generator_input)
discriminator_layers_for_generator = discriminator(generator_layers)
generator_model = Model(inputs=[generator_input], outputs=[discriminator_layers_for_generator])
# We use the Adam paramaters from Gulrajani et al.
generator_model.compile(optimizer=Adam(0.0001, beta_1=0.5, beta_2=0.9), loss=wasserstein_loss)

for layer in discriminator.layers:
    layer.trainable = True

for layer in generator.layers:
    layer.trainable = False

discriminator.trainable = True
generator.trainable = False

# The discriminator_model is more complex. It takes both real image samples and random
# noise seeds as input. The noise seed is run through the generator model to get
# generated images. Both real and generated images are then run through the
# discriminator. Although we could concatenate the real and generated images into a
# single tensor, we don't (see model compilation for why).
real_samples = Input(shape=images.shape[1:])
generator_input_for_discriminator = Input(shape=(INPUT_LEN,))
generated_samples_for_discriminator = generator(generator_input_for_discriminator)
discriminator_output_from_generator = discriminator(generated_samples_for_discriminator)
discriminator_output_from_real_samples = discriminator(real_samples)


averaged_samples = RandomWeightedAverage()([real_samples,
                                            generated_samples_for_discriminator])
# We then run these samples through the discriminator as well. Note that we never
# really use the discriminator output for these samples - we're only running them to
# get the gradient norm for the gradient penalty loss.
averaged_samples_out = discriminator(averaged_samples)


partial_gp_loss = partial(gradient_penalty_loss,
                            averaged_samples=averaged_samples, gradient_penalty_weight=GRADIENT_PENALTY_WEIGHT)
# Functions need names or Keras will throw an error
partial_gp_loss.__name__ = 'gradient_penalty'


discriminator_model = Model(inputs=[real_samples, generator_input_for_discriminator],
                            outputs=[discriminator_output_from_real_samples, discriminator_output_from_generator, averaged_samples_out])


discriminator_model.compile(optimizer=Adam(0.0001, beta_1=0.5, beta_2=0.9),
                            loss=[wasserstein_loss, wasserstein_loss, partial_gp_loss])

# We make three label vectors for training. positive_y is the label vector for real
# samples, with value 1. negative_y is the label vector for generated samples, with
# value -1. The dummy_y vector is passed to the gradient_penalty loss function and
# is not used.
positive_y = np.ones((BATCH_SIZE, 1), dtype=np.float32)
negative_y = -positive_y
dummy_y = np.zeros((BATCH_SIZE, 1), dtype=np.float32)

intervado_guardado = 50



for epoch in range(EPOCHS):
    start = time()
    np.random.shuffle(images)
    print("Epoch: ", epoch)
    print("Number of batches: ", int(n_images // BATCH_SIZE))
    discriminator_loss = []
    generator_loss = []
    minibatches_size = BATCH_SIZE * TRAINING_RATIO
    print('Tenemos ', int(n_images // (BATCH_SIZE * TRAINING_RATIO)), ' minibatches.')
    for i in range(int(n_images // (BATCH_SIZE * TRAINING_RATIO))):
        discriminator_minibatches = images[i * minibatches_size: (i + 1) * minibatches_size]

        for j in range(TRAINING_RATIO):
            image_batch = discriminator_minibatches[j * BATCH_SIZE: (j + 1) * BATCH_SIZE]
            noise = np.random.normal(0, 1, (BATCH_SIZE, INPUT_LEN)).astype(np.float32)
            # noise = np.random.uniform(-1,1,(BATCH_SIZE, INPUT_LEN)).astype(np.float32)
            discriminator_loss_val = discriminator_model.train_on_batch([image_batch, noise],
                                                                        [positive_y, negative_y, dummy_y])
            discriminator_loss.append(discriminator_loss_val)

        # generator_loss_val = generator_model.train_on_batch(np.random.uniform(-1,1,(BATCH_SIZE, INPUT_LEN)), positive_y)
        generator_loss_val = generator_model.train_on_batch(np.random.normal(0, 1, (BATCH_SIZE, INPUT_LEN)), positive_y)
        generator_loss.append(generator_loss_val)

    if epoch % intervado_guardado == 0:
        base_path = os.getcwd()
        generator.save_weights(os.path.join("weights", "generator_epoch_" + str(epoch) + ".h5"))
        discriminator.save_weights(os.path.join("weights", "discriminator_epoch_" + str(epoch) + ".h5"))
        imgs_path = os.path.join(base_path, "imgs")
        save_imgs(generator, discriminator, imgs_path, epoch, (INPUT_LEN,))


