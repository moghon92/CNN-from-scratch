# Vectorized Implementation of CNN Classifier

This project presents a vectorized implementation of a Convolutional Neural Network (CNN) classifier from scratch. The implementation includes various components commonly found in CNN architectures, such as convolutional layers, linear layers, max pooling layers, ReLU activation, and softmax cross-entropy loss.

## Contents

- [Overview](#overview)
- [Usage](#usage)

## Overview

A CNN is a deep learning architecture primarily used for image classification tasks. It consists of multiple layers, including convolutional layers for feature extraction and transformation, pooling layers for spatial downsampling, linear layers for classification, and activation functions for introducing non-linearity.

This implementation provides vectorized versions of key CNN components, which enhance performance and computation efficiency. The components include:
- `convolution.py`: Implements the convolutional layer, convolving the input with multiple filters across all channels.
- `Linear.py`: Represents a linear layer with weight `W` and bias `b`, computing the output as `y = Wx + b`.
- `Maxpool.py`: Implements the vectorized max pooling layer for downsampling the input.
- `RelU.py`: Provides the ReLU activation function from scratch.
- `Softmax_ce.py`: Implements the softmax activation function and cross-entropy loss from scratch.

## Usage

To utilize the vectorized CNN classifier implementation, follow these steps:
1. Define your CNN architecture by instantiating the required layers (e.g., convolutional, linear, pooling) and activation functions.
2. Implement the forward pass by chaining the layers together, passing the input through each layer.
3. Compute the loss using the softmax activation function and cross-entropy loss.
4. Perform backpropagation to update the parameters (weights and biases) of the model.
5. Repeat steps 4-6 for a specified number of epochs to train the model on your dataset.



Note: This implementation is for educational purposes and may not be as optimized or feature-complete as existing deep learning frameworks.
