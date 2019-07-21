from keras import layers
from keras import models
from keras import optimizers
import cv2
import os
import numpy as np
import importlib
from augmentation import utils


class MotherGAN:

    def __init__(self, generator, discriminator, combined, img_shape, noise_shape, clip_value=0.01):

        self.img_shape = img_shape
        self.noise_shape = noise_shape

        self.generator = generator
        self.discriminator = discriminator
        self.combined = combined

        self.clip_value = clip_value

    def train(self, real_images, base_path, training_name, epochs=2000,
              batch_size=100, save_interval=100, clip_weights=False):

        imgs_path, model_path = self.generate_folders(base_path, training_name)
        half_batch = batch_size//2

        for epoch in range(epochs):
            idx_batches = utils.make_indices_groups(real_images, size_group=half_batch)
            n_batches = len(idx_batches)
            for i, batch_idx in enumerate(idx_batches):

                batch_images = real_images[batch_idx]
                noise = np.random.normal(0, 1, (half_batch, *self.noise_shape))

                # Generate a half batch of new images
                generated_imgs = self.generator.predict(noise)

                # Train the discriminator
                self.discriminator.trainable = True
                d_loss_real = self.discriminator.train_on_batch(batch_images, np.ones((half_batch, 1)))
                d_loss_fake = self.discriminator.train_on_batch(generated_imgs, np.zeros((half_batch, 1)))
                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

                if clip_weights:
                    # Clip discriminator weights
                    for l in self.discriminator.layers:
                        weights = l.get_weights()
                        weights = [np.clip(w, -self.clip_value, self.clip_value) for w in weights]
                        l.set_weights(weights)

                noise = np.random.normal(0, 1, (batch_size, *self.noise_shape))

                # The generator wants the discriminator to label the generated samples
                # as valid (ones)
                valid_y = np.array([1] * batch_size)
                self.discriminator.trainable = False
                # Train the generator
                g_loss = self.combined.train_on_batch(noise, valid_y)

                # Plot the progress
                print("Epoch %d/%d, batch %d/%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, epochs, i + 1,
                                                                                             n_batches, d_loss[0],
                                                                                             100*d_loss[1], g_loss))
                # If at save interval => save generated image samples
                if epoch % save_interval == 0:
                    self.save_imgs(imgs_path, epoch)

        self.save_all_models(model_path)


    def save_all_models(self, model_path):

        self.generator.save(os.path.join(model_path, "generator.h5"))
        self.discriminator.save(os.path.join(model_path, "discriminator.h5"))
        self.combined.save(os.path.join(model_path, "combined.h5"))

    def save_imgs(self, imgs_path, epoch, n_imgs=5):

        noise = np.random.normal(0, 1, (n_imgs , *self.noise_shape))
        gen_imgs = self.generator.predict(noise)

        # Rescale images 0 - 1
        gen_img = 0.5 * gen_imgs + 0.5
        for i in range(n_imgs):
            img_name = "generated_img_%d_epoch_%d.png" % (i, epoch)
            re_scaled = (gen_img - np.min(gen_img)) * 255 / (np.max(gen_img) - np.min(gen_img))
            cv2.imwrite(os.path.join(imgs_path, img_name),
                        np.concatenate([re_scaled[i, :, :, 0], re_scaled[i, :, :, 1]], axis=1)*255)


    def generate_folders(self, base_path, training_name):

        imgs_path = os.path.join(base_path, "augmentation", "generated_imgs", training_name)
        if not os.path.exists(imgs_path):
            os.mkdir(imgs_path)

        model_path = os.path.join(base_path, "augmentation", "gan_models", training_name)
        if not os.path.exists(model_path):
            os.mkdir(model_path)

        return imgs_path, model_path


