import tensorflow as tf
import numpy as np
import os


# This script may be useful to make predictions on a test set.
# It shows how to load the trained model and has simple routines to load
# and process a test set, as well as to make predictions on it.

####
load_model = tf.keras.models.load_model
loaded_model = load_model('fontclassifier.h5')

# predict on one new image
image = tf.keras.preprocessing.image
test_image = image.load_img('Other/elegant-Playball.png', target_size=(64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)
result = loaded_model.predict(test_image)

# predict on a test set


def process_test_set(folder):
    """ simple function to load images and process a set of images into a format suitable to be fed to the model
    input:
        -folder: the name of the folder in which the images are located
    output:
        - a list of keras preprocessed images
    """
    test_set = []
    for entry in os.scandir(os.getcwd() + '/' + folder):
        test_image = image.load_img(
            folder +
            '/' +
            entry.name,
            target_size=(
                64,
                64))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis=0)
        test_set.append(test_image)

    return test_set


def get_predictions(test_set):
    """ simple function to obtain the set of predictions on a test set
    input:
        -test_set: a list of keras preprocessed images
    output:
        - An array of predictions
    """
    predictions = []
    for entry in test_set:
        predictions.append(loaded_model.predict(entry))

    return np.array(predictions)


test_set = process_test_set('test_set/sansserif')
# predicting the sansserif images of the test set
predictions = get_predictions(test_set)
# computing the accuracy of the model on sansserif images
accuracy_sans_serif = np.sum(predictions == 0) / len(predictions)

test_set = process_test_set('test_set/serif')
# predicting the serif images of the test set
predictions = get_predictions(test_set)
# computing the accuracy of the model on serif images
accuracy_serif = np.sum(predictions == 1) / len(predictions)

print("The accuracy on the sans serif test set is: " + str(accuracy_sans_serif))

print("The accuracy on the serif test set is: " + str(accuracy_serif))
