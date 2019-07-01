import numpy as np
from augmentation import discriminator
from augmentation import generator
from keras import layers
from keras import models
from keras import optimizers
import keras.backend as K

class GenericGAN:

    def __init__(self, img_shape):
        self.img_shape = img_shape

        optimizer = optimizers.Adam(0.0002, 0.5)

        # Build and compile the discriminator
        self.discriminator = discriminator.create_model(self.img_shape)

        # Build and compile the generator
        self.generator = generator.create_model(self.img_shape)

        # The generator takes noise as input and generated imgs
        z = layers.Input(shape=self.img_shape)
        img = self.generator(z)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The valid takes generated images as input and determines validity
        valid = self.discriminator(img)

        # The combined model  (stacked generator and discriminator) takes
        # noise as input => generates images => determines validity
        self.combined = models.Model(z, valid)
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)


    def train(self, X, y, base_path, epochs=10, batch_size=32, save_interval=200):

        images = np.concatenate([X, y], axis=3)
        half_batch = batch_size//2

        for epoch in range(epochs):

            # Select a random half batch of images
            idxs = np.random.randint(0, X.shape[0], half_batch)
            batch_images = images[idxs]

            noise = np.random.normal(0, 1, (half_batch, 512))

            # Generate a half batch of new images
            generated_imgs = self.generator.predict(noise)

            # Train the discriminator
            d_loss_real = self.discriminator.train_on_batch(batch_images, np.ones((half_batch, 1)))
            d_loss_fake = self.discriminator.train_on_batch(generated_imgs, np.zeros((half_batch, 1)))
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            noise = np.random.normal(0, 1, (batch_size, 100))

            # The generator wants the discriminator to label the generated samples
            # as valid (ones)
            valid_y = np.array([1] * batch_size)

            # Train the generator
            g_loss = self.combined.train_on_batch(noise, valid_y)

            # Plot the progress
            print("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100 * d_loss[1], g_loss))

            # If at save interval => save generated image samples
            if epoch % save_interval == 0:
                self.save_imgs(epoch)

    def save_imgs(self, epoch):
        """
        r, c = 5, 5
        noise = np.random.normal(0, 1, (r * c, 100))
        gen_imgs = self.generator.predict(noise)

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i, j].imshow(gen_imgs[cnt, :, :, 0], cmap='gray')
                axs[i, j].axis('off')
                cnt += 1
        fig.savefig("gan/images/mnist_%d.png" % epoch)
        plt.close()
        """
        pass

