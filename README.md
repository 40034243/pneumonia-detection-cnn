# Pneumonia Detection CNN
This GitHub repository contains code for a convolutional neural network (CNN) model that performs binary classification on chest X-ray images. The purpose of this code is to train a model to classify X-ray images into two categories: normal and abnormal.

The code uses the Keras library, which is a high-level neural networks API running on top of TensorFlow. It imports necessary modules and packages, including os, numpy, and Keras modules such as Sequential, Dense, Dropout, Flatten, Conv2D, MaxPooling2D, ImageDataGenerator, and Adam optimizer.

The code begins by defining the directories for training, validation, and testing data. It then sets up an ImageDataGenerator for data augmentation on the training data, which includes rescaling, shear range, zoom range, and horizontal flipping. Another ImageDataGenerator is created for the validation and test data, but without data augmentation.

Next, the code defines the architecture of the CNN model using the Sequential API. The model consists of two convolutional layers with ReLU activation, followed by max pooling and dropout regularization. The tensor output from the convolutional layers is flattened and fed into fully connected layers with ReLU activation and dropout regularization. The final layer uses a sigmoid activation function for binary classification.

The model is then compiled with a binary cross-entropy loss function, the Adam optimizer, and accuracy as the evaluation metric.

The model is trained using the fit_generator function, which iterates over the training data generator for a specified number of epochs. The validation data generator is used for monitoring the model's performance during training.

After training, the model is evaluated on the test data using the evaluate_generator function, calculating the test loss and accuracy.

The code outputs the test loss and test accuracy to the console.

Overall, this code provides a complete pipeline for training and evaluating a CNN model on chest X-ray images, with the goal of classifying them as normal or abnormal.
