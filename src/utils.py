import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
import os

def save_images(generator, epoch, examples=25, dim=(5, 5), figsize=(10, 10)):
    """
    Save generated images from the generator model.
    
    Args:
        generator: Trained generator model
        epoch: Current training epoch (for filename)
        examples: Number of images to generate
        dim: Grid dimensions for display
        figsize: Size of the output figure
    """
    noise = np.random.normal(0, 1, (examples, 100))
    generated_images = generator.predict(noise)
    generated_images = 0.5 * generated_images + 0.5  # Rescale from [-1,1] to [0,1]
    
    plt.figure(figsize=figsize)
    for i in range(examples):
        plt.subplot(dim[0], dim[1], i+1)
        plt.imshow(generated_images[i, :, :, 0], cmap='gray')
        plt.axis('off')
    plt.tight_layout()
    
    # Create outputs directory if it doesn't exist
    os.makedirs("outputs", exist_ok=True)
    plt.savefig(f"outputs/gan_image_epoch_{epoch}.png")
    plt.close()

def load_data(normalize=True, expand_dims=True):
    """
    Load and preprocess MNIST dataset.
    
    Args:
        normalize: Whether to normalize pixel values to [-1, 1]
        expand_dims: Whether to add channel dimension
        
    Returns:
        Processed MNIST images
    """
    (X_train, _), (_, _) = mnist.load_data()
    
    if normalize:
        X_train = (X_train.astype(np.float32) - 127.5) / 127.5  # Normalize to [-1, 1]
    
    if expand_dims:
        X_train = np.expand_dims(X_train, axis=3)  # Add channel dimension
    
    return X_train

def generate_noise(batch_size, noise_dim=100):
    """
    Generate random noise for generator input.
    
    Args:
        batch_size: Number of noise vectors to generate
        noise_dim: Dimension of each noise vector
        
    Returns:
        Array of random noise vectors
    """
    return np.random.normal(0, 1, (batch_size, noise_dim))

def plot_training_history(g_losses, d_losses, d_accuracies):
    """
    Plot training history of GAN.
    
    Args:
        g_losses: List of generator losses
        d_losses: List of discriminator losses
        d_accuracies: List of discriminator accuracies
    """
    plt.figure(figsize=(12, 6))
    
    # Plot losses
    plt.subplot(1, 2, 1)
    plt.plot(g_losses, label='Generator Loss')
    plt.plot(d_losses, label='Discriminator Loss')
    plt.title('Training Losses')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(d_accuracies, label='Discriminator Accuracy')
    plt.title('Discriminator Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig("outputs/training_history.png")
    plt.close()

def create_gif(image_dir="outputs", output_file="training_progress.gif"):
    """
    Create a GIF from generated images (requires imageio).
    
    Args:
        image_dir: Directory containing generated images
        output_file: Name of output GIF file
    """
    try:
        import imageio
        from PIL import Image
    except ImportError:
        print("Please install imageio and Pillow to create GIFs")
        return
    
    images = []
    for file_name in sorted(os.listdir(image_dir)):
        if file_name.startswith("gan_image_epoch_") and file_name.endswith(".png"):
            file_path = os.path.join(image_dir, file_name)
            images.append(imageio.imread(file_path))
    
    if images:
        imageio.mimsave(os.path.join(image_dir, output_file), images, duration=0.5)
        print(f"GIF saved to {os.path.join(image_dir, output_file)}")
    else:
        print("No images found to create GIF")