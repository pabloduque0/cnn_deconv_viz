import numpy as np
from augmentation.discriminators import wasserstein_discriminator
from augmentation.generators import wasserstein_generator
from keras import layers
from keras import models
from keras import optimizers
from augmentation import metrics
from keras import losses
from keras import optimizers
from augmentation.mothergan import MotherGAN
from augmentation import utils


class WassersteinGAN(MotherGAN):

    def __init__(self, img_shape, noise_shape, n_discriminator=8):

        discriminator_model = wasserstein_discriminator.create_model(img_shape)
        # Build and compile the discriminator
        discriminator_model.compile(optimizers.Adam(lr=0.0001, beta_1=0.5, beta_2=0.99),
                                    loss=metrics.wasserstein_loss, metrics=['accuracy'])

        # Build and compile the generator
        generator_model = wasserstein_generator.create_model(noise_shape)

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
        combined_model.compile(optimizers.Adam(lr=0.0001, beta_1=0.5, beta_2=0.99),
                               loss=metrics.wasserstein_loss,
                               metrics=['accuracy'])
        self.n_discriminator = n_discriminator

        super().__init__(generator_model, discriminator_model, combined_model, img_shape, noise_shape)



    def train(self, real_images, base_path, training_name, epochs=5000,
              batch_size=16, save_interval=50):

        imgs_path, model_path = self.generate_folders(base_path, training_name)
        half_batch = batch_size//2

        for epoch in range(epochs):
            idx_batches = utils.make_indices_groups(real_images, size_group=half_batch*self.n_discriminator)
            n_batches = len(idx_batches)
            for i, batch_idx in enumerate(idx_batches):

                batch_images = real_images[batch_idx]
                idx_sub_batches = utils.make_indices_groups(batch_images,
                                                        size_group=half_batch)

                for j in idx_sub_batches:
                    sub_batch = batch_images[j]
                    noise = np.random.normal(0, 1, (half_batch, *self.noise_shape))
                    # Generate a half batch of new images
                    generated_imgs = self.generator.predict(noise)

                    # Train the discriminator
                    self.discriminator.trainable = True
                    d_loss_real = self.discriminator.train_on_batch(sub_batch, -np.ones((half_batch, 1)))
                    d_loss_fake = self.discriminator.train_on_batch(generated_imgs, np.ones((half_batch, 1)))
                    d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

                    # Clip discriminator weights
                    for l in self.discriminator.layers:
                        weights = l.get_weights()
                        weights = [np.clip(w, -self.clip_value, self.clip_value) for w in weights]
                        l.set_weights(weights)

                noise = np.random.normal(0, 1, (batch_size, *self.noise_shape))

                # The generator wants the discriminator to label the generated samples
                # as valid (ones)
                valid_y = np.array([-1] * batch_size)
                self.discriminator.trainable = False
                # Train the generator
                g_loss = self.combined.train_on_batch(noise, valid_y)

                # Plot the progress
                print("Epoch %d/%d, batch %d/%d [D loss: %f] [G loss: %f]" % (epoch, epochs, i + 1,
                                                                              n_batches, 1 - d_loss[0],
                                                                              1 - g_loss[0]))
                # If at save interval => save generated image samples
                if epoch % save_interval == 0:
                    self.save_imgs(imgs_path, epoch)

        self.save_all_models(model_path)

