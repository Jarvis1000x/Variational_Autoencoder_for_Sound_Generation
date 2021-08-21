# Variational Autoencoder for Sound Generation
This repository is a neural network architecture for generating sound using Variational Autoencoders. The encoder and decoder components of the VAE are made using Convolutional Neural Net. The audio is first converted into spectrograms, and then fed to the network. The network then gives a spectogram as well, which is converted back into Audio for the final output.

## Dataset

This Variational Autoencoder was trained on the [Free Spoken Digit Dataset (FSDD)](https://github.com/Jakobovski/free-spoken-digit-dataset) to produce sounds of spoken digits. However, it can also be trained on other such sound datasets like [Nsynth](https://magenta.tensorflow.org/datasets/nsynth) and [MAESTRO](https://magenta.tensorflow.org/datasets/maestro) by google to produce musical notes.
