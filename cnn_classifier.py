from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import os


# Script used to build and train the model.
# Please if h5py not available, download it using 'pip install h5py'. This package is required in order to be able to load the trained model.
# To directly make predictions on this model please see @ makePredictions.py

current_wd = os.getcwd()

# using Keras image_data_generator to process and load the data so that we can use it
# This class also allows us to generate new data (augment the dataset) for
# training the model via rotations, zooming, scaling, whitening, and other
# transformation techniques.
image_data_generator = tf.keras.preprocessing.image.ImageDataGenerator


train_datagen = image_data_generator(rescale=1. / 255,
                                     shear_range=0.2,
                                     zoom_range=0.2,
                                     horizontal_flip=True)

eval_datagen = image_data_generator(rescale=1. / 255)

training_set = train_datagen.flow_from_directory(
    'training_set',
    target_size=(
        64,
        64),
    batch_size=32,
    class_mode='binary',
    save_to_dir='augmented/training',
    save_prefix='aug',
    save_format='png')

eval_set = eval_datagen.flow_from_directory(
    'eval_set',
    target_size=(
        64,
        64),
    batch_size=32,
    class_mode='binary',
    save_to_dir='augmented/eval',
    save_prefix='aug',
    save_format='png')


#################

# Building the model

sequential = tf.keras.models.Sequential
conv2d = tf.keras.layers.Conv2D
maxpooling2d = tf.keras.layers.MaxPooling2D
flatten = tf.keras.layers.Flatten
dense = tf.keras.layers.Dense
dropout = tf.keras.layers.Dropout

# 0-Intialize a sequential model such that each layer is sequentially
# connected to the next layer
classifier = sequential()

# 1-Convolutional layer
# Convolution layer with 32 filters, 5 * 5 kernel,layer output of
# dimension 64*64*3 and relu activation
classifier.add(conv2d(32, (5, 5), input_shape=(64, 64, 3), activation='relu'))

# 2-Pooling layer
classifier.add(maxpooling2d(pool_size=(2, 2), strides=2))

# 3-Convolutional layer
# Convolution layer with 64 filters, 5 * 5 kernel,output of implicit
# dimension and relu activation
classifier.add(conv2d(64, (5, 5), activation='relu'))

# 4- Pooling layer
classifier.add(maxpooling2d(pool_size=(2, 2), strides=2))

# 5-flattening
classifier.add(flatten())

# 6-Fully Connected Layer
classifier.add(dense(units=128, activation='relu'))
# dropout rate indicates percentage of units that should be randomly dropped.
classifier.add(dropout(rate=0.5))
# Final layer outputs a class for each sample. Uses sigmoid function
classifier.add(dense(units=1, activation='sigmoid'))

# 7-Compile the model
# We use adam to optimize the parameters, binary crossentropy as our loss
# function and we evaluate the model on its accuracy
classifier.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy'])

# training the model on the training set
# This step also computes the validation accuracy
# number of validation steps, number of epochsm and number of steps per
# epoch can be increased to improve accuracy, I am leaving it as is to
# decrease the running time.
classifier.fit_generator(
    training_set,
    steps_per_epoch=2000,
    epochs=1,
    validation_data=eval_set,
    validation_steps=100)

# saving the classifier as 'fontclassifier.h5' so that we can load it
# later and make predictions on new data
classifier.save('fontclassifier.h5')
