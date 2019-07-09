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

class GenericGAN:

    def __init__(self, img_shape):
        self.img_shape = img_shape

        # Build and compile the discriminator
        self.discriminator = discriminator.create_model(self.img_shape)
        self.discriminator.compile(optimizers.Adam(lr=0.00001, beta_1=0, beta_2=0.99),
                                   loss=losses.binary_crossentropy, metrics=['accuracy'])

        # Build and compile the generator
        self.generator = generator.create_model((12, 12, 2))
        self.generator.compile(optimizers.Adam(lr=0.00001, beta_1=0, beta_2=0.99),
                               loss=losses.binary_crossentropy)

        # The generator takes noise as input and generated imgs
        z = layers.Input(shape=(12, 12, 2))
        img = self.generator(z)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False
        # The valid takes generated images as input and determines validity
        valid = self.discriminator(img)

        # The combined model  (stacked generator and discriminator) takes
        # noise as input => generates images => determines validity
        self.combined = models.Model(z, valid)
        self.combined.compile(optimizers.Adam(lr=0.00001, beta_1=0, beta_2=0.99), loss=metrics.generator_loss)


    def train(self, X, y, base_path, training_name, epochs=400, batch_size=32, save_interval=25):

        imgs_path, model_path = self.generate_folders(base_path, training_name)

        images = X
        half_batch = batch_size//2

        for epoch in range(epochs):

            # Select a random half batch of images
            idxs = np.random.randint(0, X.shape[0], half_batch)
            batch_images = images[idxs]

            noise = np.random.normal(0, 1, (half_batch, 12, 12, self.img_shape[-1]))

            # Generate a half batch of new images
            generated_imgs = self.generator.predict(noise)

            # Train the discriminator
            self.discriminator.trainable = True
            d_loss_real = self.discriminator.train_on_batch(batch_images, np.ones((half_batch, 1)))
            d_loss_fake = self.discriminator.train_on_batch(generated_imgs, np.zeros((half_batch, 1)))
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            noise = np.random.normal(0, 1, (batch_size, *(12, 12, 2)))

            # The generator wants the discriminator to label the generated samples
            # as valid (ones)
            valid_y = np.array([1] * batch_size)
            self.discriminator.trainable = False
            # Train the generator
            g_loss = self.combined.train_on_batch(noise, valid_y)

            # Plot the progress
            print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))

            # If at save interval => save generated image samples
            if epoch % save_interval == 0:
                self.save_imgs(imgs_path, epoch)

        self.save_all_models(model_path)


    def save_all_models(self, model_path):

        self.generator.save(os.path.join(model_path, "generator.h5"))
        self.discriminator.save(os.path.join(model_path, "discriminator.h5"))
        self.combined.save(os.path.join(model_path, "combined.h5"))

    def save_imgs(self, imgs_path, epoch, n_imgs=5):

        noise = np.random.normal(0, 1, (n_imgs , *(12, 12, 2)))
        gen_imgs = self.generator.predict(noise)

        # Rescale images 0 - 1
        gen_img = 0.5 * gen_imgs + 0.5
        for i in range(n_imgs):
            img_name = "generated_img_%d_epoch_%d.png" % (i, epoch)
            cv2.imwrite(os.path.join(imgs_path, img_name),
                        np.concatenate([gen_img[i, :, :, 0], gen_img[i, :, :, 1]], axis=1))


    def generate_folders(self, base_path, training_name):

        imgs_path = os.path.join(base_path, "generated_imgs", training_name)
        if not os.path.exists(imgs_path):
            os.mkdir(imgs_path)

        model_path = os.path.join(base_path, "gan_models", training_name)
        if not os.path.exists(model_path):
            os.mkdir(model_path)

        return imgs_path, model_path
