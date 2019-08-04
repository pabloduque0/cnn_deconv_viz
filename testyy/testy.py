
import numpy as np
import cv2
from keras.models import Model, Sequential
from keras.layers import Input, Dense, Reshape, Flatten
from keras.layers.convolutional import Convolution2D, Conv2DTranspose
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
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
import matplotlib.pyplot as plt
from functools import partial
import os

try:
    from PIL import Image
except ImportError:
    print('This script depends on pillow! Please install it (e.g. with pip install pillow)')

BATCH_SIZE = 8
# The training ratio is the number of discriminator updates
# per generator update. The paper uses 5.
GRADIENT_PENALTY_WEIGHT = 10  # As per the paper

def write_log(callback, names, logs, batch_no):
    for name, value in zip(names, logs):
        summary = tf.Summary()
        summary_value = summary.value.add()
        summary_value.simple_value = value
        summary_value.tag = name
        callback.writer.add_summary(summary, batch_no)
        callback.writer.flush()


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


def generate_images(generator_model, output_dir, epoch, n_images, method='FLAIR'):
    """Feeds random seeds into the generator and tiles and saves the output to a PNG
    file."""
    images = generator_model.predict(np.random.rand(n_images, 128))
    cols = 2

    fig = plt.figure()
    for n, image in enumerate(images):
        a = fig.add_subplot(np.ceil(n_images / float(cols)), cols, n + 1)
        if method == 'T1':
            plt.imshow(image[..., 0], cmap='gray')
        else:
            plt.imshow(image[..., 1], cmap='gray')

    fig.set_size_inches(np.array(fig.get_size_inches()) * n_images)
    fig.savefig(output_dir + str(method) + "/epoch_" + str(epoch) + ".png")
    plt.close(fig)

def sample_best_images(generator_model, discriminator, output_dir, epoch='No', n_images = 10, n_images_total = 100, flag = [1]):
    """Coger las mejores imágenes.
    Cogemos por defecto 100 imágenes del generador (controlándolo con n_images_total)
    Seleccionamos las n_images mejores y las guardamos en un archivo. Guardamos por separado las
    FLAIR y las T1.
    """
    images = generator_model.predict(np.random.rand(n_images_total, 128))
    images_mark = discriminator.predict(images).reshape((n_images_total))
    order = np.argsort(-images_mark)[:n_images]

    cols = 2
    images_final = images[order,...]
    print(images.shape)


    if 1 in flag:
        figFLAIR = plt.figure()
        for n, image in enumerate(images_final):
            a = figFLAIR.add_subplot(np.ceil(n_images / float(cols)), cols, n + 1)
            plt.imshow(image[..., 1], cmap='gray')
        if not os.path.isdir(output_dir + "FLAIR/"):
            os.mkdir(output_dir + "FLAIR/")

        figFLAIR.set_size_inches(np.array(figFLAIR.get_size_inches()) * n_images)
        figFLAIR.savefig(output_dir + "FLAIR/epoch_" + str(epoch) + ".png")
        plt.close(figFLAIR)

        figT1 = plt.figure()
        for n, image in enumerate(images_final):
            a = figT1.add_subplot(np.ceil(n_images / float(cols)), cols, n + 1)
            plt.imshow(image[..., 0], cmap='gray')

        if not os.path.isdir(output_dir + "T1/"):
            os.mkdir(output_dir + "T1/")

        figT1.set_size_inches(np.array(figT1.get_size_inches()) * n_images)
        print(output_dir + "T1/epoch_" + str(epoch) + ".png")
        figT1.savefig(output_dir + "T1/epoch_" + str(epoch) + ".png")
        plt.close(figT1)

        figMask = plt.figure()
        for n, image in enumerate(images_final):
            a = figMask.add_subplot(np.ceil(n_images / float(cols)), cols, n + 1)
            plt.imshow(image[..., 2], cmap='gray')

        if not os.path.isdir(output_dir + "Mask/"):
            os.mkdir(output_dir + "Mask/")

        figMask.set_size_inches(np.array(figMask.get_size_inches()) * n_images)
        print(output_dir + "Mask/epoch_" + str(epoch) + ".png")
        figMask.savefig(output_dir + "Mask/epoch_" + str(epoch) + ".png")
        plt.close(figMask)
    elif 2 in flag:
        fig = np.empty((3,256, 256,1))
        fig[0,...] = images_final[...,0].reshape((1 ,256, 256, 1))
        fig[1,...] = images_final[...,1].reshape((1 ,256, 256, 1))
        fig[2,...] = images_final[...,2].reshape((1 ,256, 256, 1))

        return fig
    elif 3 in flag:
        if not os.path.isdir(output_dir + "Img_Grandes/"):
            os.mkdir(output_dir + "Img_Grandes/")
            os.mkdir(output_dir + "Img_Grandes/FLAIR")
            os.mkdir(output_dir + "Img_Grandes/T1")
            os.mkdir(output_dir + "Img_Grandes/MASK")
        images_final_2 = images_final[:3, ...]
        for n, image in enumerate(images_final_2):
            imFLAIR = plt.figure()
            plt.imshow(image[...,1], cmap = 'gray')
            imFLAIR.savefig(output_dir + "Img_Grandes/FLAIR/epoch_" + str(epoch) + "_n" + str(n))
            plt.close(imFLAIR)

            imT1 = plt.figure()
            plt.imshow(image[...,0], cmap = 'gray')
            imFLAIR.savefig(output_dir + "Img_Grandes/T1/epoch" + str(epoch) + "_n" + str(n))
            plt.close(imT1)

            imMASK= plt.figure()
            plt.imshow(image[...,2], cmap = 'gray')
            imFLAIR.savefig(output_dir + "Img_Grandes/MASK/epoch" + str(epoch) + "_n" + str(n))
            plt.close(imMASK)


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
                    np.concatenate([re_scaled[:, :, 0], re_scaled[:, :, 1], re_scaled[:, :, 2]], axis=1))



EPOCHS = 20000
BATCH_SIZE = 8
# The training ratio is the number of discriminator updates
# per generator update. The paper uses 5.
TRAINING_RATIO = 8
GRADIENT_PENALTY_WEIGHT = 10  # As per the paper
INPUT_LEN = 128
output_dir = 'output/'
discriminator_weights = 'Weights/discriminator_epoch_10600.h5'
generator_weights = 'Weights/generator_epoch_10600.h5'
muestra = None # Si queremos coger una muestra de las imágenes. None para no utilizar
intervalo_guardado = 50
file_discriminator = output_dir + 'LOSS/disc_loss.txt'
file_generator = output_dir + 'LOSS/gen_loss.txt'
log_path = 'logs/'

imagenes_muestra = None  # None para leer el completo

kernel_size_generator = 4
initial_epoch = 0
final_epoch = initial_epoch + EPOCHS - 1


def make_discriminator():
    """Creates a discriminator model that takes an image as input and outputs a single
    value, representing whether the input is real or generated. Unlike normal GANs, the
    output is not sigmoid and does not represent a probability! Instead, the output
    should be as large and negative as possible for generated inputs and as large and
    positive as possible for real inputs.
    Note that the improved WGAN paper suggests that BatchNormalization should not be
    used in the discriminator."""

    model = Sequential()
    model.add(Convolution2D(32, 5, padding='same', strides=[2, 2], input_shape=(256, 256, 3)))
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

    model.add(Dense(4 * 4 * 2048, input_dim=INPUT_LEN))
    # model.add(BatchNormalization())
    model.add(LeakyReLU())
    model.add(Reshape((4, 4, 2048), input_shape=(4 * 4 * 2048,)))
    bn_axis = -1

    model.add(Conv2DTranspose(1024, kernel_size_generator, strides=2, padding='same'))
    # model.add(BatchNormalization(axis=bn_axis))
    model.add(LeakyReLU())

    model.add(Conv2DTranspose(512, kernel_size_generator, strides=2, padding='same'))
    # model.add(BatchNormalization(axis=bn_axis))
    model.add(LeakyReLU())

    model.add(Conv2DTranspose(256, kernel_size_generator, strides=2, padding='same'))
    # model.add(BatchNormalization(axis=bn_axis))
    model.add(LeakyReLU())

    model.add(Conv2DTranspose(128, kernel_size_generator, strides=2, padding='same'))
    # model.add(BatchNormalization(axis=bn_axis))
    model.add(LeakyReLU())

    model.add(Conv2DTranspose(64, kernel_size_generator, strides=2, padding='same'))
    # model.add(BatchNormalization(axis=bn_axis))
    model.add(LeakyReLU())

    model.add(Conv2DTranspose(3, kernel_size_generator, strides=2, padding='same', activation='tanh'))
    # El output de esta última es 256x256x3

    return model


images = np.load('../../muestra_seleccionada.npy')

images = [2. * (image - np.min(image)) / np.ptp(image) - 1 for image in images]

# El generador toma imágenes 256x256x3. Como las tenemos 200x200, hay que redimensionarlas:
dim_final = (256, 256)
images = np.array([cv2.resize(image, dim_final, interpolation=cv2.INTER_AREA) for image in images])

n_images = images.shape[0]
print(n_images)

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

n_minibatches = int(n_images // (BATCH_SIZE * TRAINING_RATIO))
minibatch_epochs = initial_epoch * n_minibatches
training_ratio_epoch = initial_epoch * n_minibatches * TRAINING_RATIO

print("Number of batches: ", int(n_images // BATCH_SIZE))
print('Tenemos ', int(n_images // (BATCH_SIZE * TRAINING_RATIO)), ' minibatches.')

callback = TensorBoard(log_path)
callback.set_model(generator_model)
# file_writer = tsummary.FileWriter(log_path)

discriminator_loss_mean = []
generator_loss_mean = []

for epoch in range(initial_epoch, final_epoch):
    start = time()
    np.random.shuffle(images)
    print("Epoch: ", epoch)

    discriminator_loss_epoch = []
    generator_loss_epoch = []

    minibatches_size = BATCH_SIZE * TRAINING_RATIO

    for i in range(int(n_images // (BATCH_SIZE * TRAINING_RATIO))):
        discriminator_minibatches = images[i * minibatches_size: (i + 1) * minibatches_size]

        for j in range(TRAINING_RATIO):
            image_batch = discriminator_minibatches[j * BATCH_SIZE: (j + 1) * BATCH_SIZE]
            noise = np.random.normal(0, 1, (BATCH_SIZE, INPUT_LEN)).astype(np.float32)
            # noise = np.random.uniform(-1,1,(BATCH_SIZE, INPUT_LEN)).astype(np.float32)
            discriminator_loss_val = discriminator_model.train_on_batch([image_batch, noise],
                                                                        [positive_y, negative_y, dummy_y])
            discriminator_loss_epoch.append(discriminator_loss_val)

            # PARA TENSORBOARD
            write_log(callback, ['d_loss', 'd_loss_real', 'd_loss_fake'], [
                discriminator_loss_val[0],
                discriminator_loss_val[1],
                discriminator_loss_val[2]
            ], training_ratio_epoch)

            training_ratio_epoch += 1

        # generator_loss_val = generator_model.train_on_batch(np.random.uniform(-1,1,(BATCH_SIZE, INPUT_LEN)), positive_y)
        generator_loss_val = generator_model.train_on_batch(np.random.normal(0, 1, (BATCH_SIZE, INPUT_LEN)), positive_y)
        generator_loss_epoch.append(generator_loss_val)

        # ESCRIBIR PARA TENSORBOARD
        write_log(callback, ['g_loss'], [generator_loss_val], minibatch_epochs)

        minibatch_epochs += 1

    generator_loss_mean.append(np.mean(generator_loss_epoch))
    discriminator_loss_mean.append(np.mean(discriminator_loss_epoch, axis=0))

    print('Epoch ' + str(epoch) + ' took ' + str(time() - start))

    if epoch % intervalo_guardado == 0:
        print('Saving weights')
        generator.save_weights('Weights/generator_epoch_' + str(epoch) + '.h5')
        discriminator.save_weights('Weights/discriminator_epoch_' + str(epoch) + '.h5')
        print('Weights saved')

        sample_best_images(generator, discriminator, output_dir, epoch, 10, flag=[1, 3])
        image_board = sample_best_images(generator, discriminator, output_dir, epoch, 1, flag=[2])

        base_path = os.getcwd()
        generator.save_weights(os.path.join("weights", "generator_epoch_" + str(epoch) + ".h5"))
        discriminator.save_weights(os.path.join("weights", "discriminator_epoch_" + str(epoch) + ".h5"))
        imgs_path = os.path.join(base_path, "imgs")
        save_imgs(generator, discriminator, imgs_path, epoch, (INPUT_LEN,))

        # with file_writer.as_default():
        #    tsummary.image('Epoch_' + str(epoch), image_board, step=0
        # tbc.save_image('Epoch_' + str(epoch), image_board)
        if epoch - intervalo_guardado % 1000 != 0:
            try:
                os.remove('./Weights/discriminator_epoch_' + str(epoch - intervalo_guardado) + '.h5')
            except:
                pass
            try:
                os.remove('./Weights/generator_epoch_' + str(epoch - intervalo_guardado) + '.h5')
            except:
                pass

        write_list(discriminator_loss_mean, file_discriminator)
        write_list(generator_loss_mean, file_generator)

        discriminator_loss_mean = []
        generator_loss_mean = []