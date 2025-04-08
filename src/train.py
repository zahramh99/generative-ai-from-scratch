import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from .gan_model import build_generator, build_discriminator, build_gan

def load_data():
    (X_train, _), (_, _) = mnist.load_data()
    X_train = (X_train.astype(np.float32) - 127.5) / 127.5
    X_train = np.expand_dims(X_train, axis=3)
    return X_train

def save_images(generator, epoch, examples=25, dim=(5,5)):
    noise = np.random.normal(0, 1, (examples, 100))
    generated_images = generator.predict(noise)
    generated_images = 0.5 * generated_images + 0.5
    
    plt.figure(figsize=dim)
    for i in range(examples):
        plt.subplot(dim[0], dim[1], i+1)
        plt.imshow(generated_images[i, :, :, 0], cmap='gray')
        plt.axis('off')
    plt.tight_layout()
    plt.savefig(f"outputs/gan_image_epoch_{epoch}.png")
    plt.close()

def train(epochs=10000, batch_size=64):
    X_train = load_data()
    
    generator = build_generator()
    discriminator = build_discriminator()
    discriminator.compile(optimizer=Adam(0.0002, 0.5), 
                         loss='binary_crossentropy', 
                         metrics=['accuracy'])
    
    gan = build_gan(generator, discriminator)
    
    real = np.ones((batch_size, 1))
    fake = np.zeros((batch_size, 1))
    
    for epoch in range(epochs):
        # Train discriminator
        idx = np.random.randint(0, X_train.shape[0], batch_size)
        real_images = X_train[idx]
        
        noise = np.random.normal(0, 1, (batch_size, 100))
        generated_images = generator.predict(noise)
        
        d_loss_real = discriminator.train_on_batch(real_images, real)
        d_loss_fake = discriminator.train_on_batch(generated_images, fake)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
        
        # Train generator
        noise = np.random.normal(0, 1, (batch_size, 100))
        g_loss = gan.train_on_batch(noise, real)
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch} [D loss: {d_loss[0]}, acc.: {100*d_loss[1]}] [G loss: {g_loss}]")
            save_images(generator, epoch)

if __name__ == "__main__":
    train()