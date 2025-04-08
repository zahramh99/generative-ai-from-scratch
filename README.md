# Generative AI Model From Scratch with Python

This project implements a Generative Adversarial Network (GAN) from scratch to generate handwritten digits using the MNIST dataset.

## Generative Adversarial Networks (GANs)
Generator: Generates new data samples.
Discriminator: Evaluates whether a given data sample is real (from the training data) or fake (generated by the generator).
The two networks are trained together in a zero-sum game: the generator tries to fool the discriminator, while the discriminator aims to accurately distinguish real from fake data.
A GAN consists of the following key components:
Noise Vector: A random input vector fed into the generator.
Generator: A neural network that transforms the noise vector into a data sample.
Discriminator: A neural network that classifies input data as real or fake.

## Project Structure
- `src/`: Contains the main implementation code
- `outputs/`: Stores generated images during training
- `notebooks/`: Contains exploratory analysis (if any)

## Requirements
- Python 3.6+
- Keras
- NumPy
- Matplotlib

## Installation
```bash
pip install -r requirements.txt

## Usage
To train the model:
python src/train.py
## Results
The model generates handwritten digits after training. Sample outputs are stored in the outputs/ directory.
