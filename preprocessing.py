from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import shutil
from PIL import Image
from numpy import array
import numpy as np

# This script is used to process the data. We shuffle and split the data
# into training set, evaluation set, and test set.


current_wd = os.getcwd()

folders = []
files = []


num_eval_examples = 300
num_test_examples = 100
np.random.seed(0)

for entry in os.scandir(current_wd):
    if 'augmented' == entry.name:
        shutil.rmtree('augmented')
    if 'test_set' == entry.name:
        shutil.rmtree('test_set')
    if 'training_set' == entry.name:
        shutil.rmtree('training_set')
    if 'eval_set' == entry.name:
        shutil.rmtree('eval_set')

os.makedirs('augmented')
os.makedirs('augmented/eval')
os.makedirs('augmented/training')

os.makedirs('test_set/sansserif')
os.makedirs('test_set/serif')
os.makedirs('training_set/sansserif')
os.makedirs('training_set/serif')
os.makedirs('eval_set/sansserif')
os.makedirs('eval_set/serif')


def get_data(path):
    """ simple routine to load the images into a dataset
    input: the path to the folder containing the images
    outputs:
        -files:A list containing the name of all the files in the folder.
        -images: A list containing the images corresponding to each file represented as a numpy array.
    """
    files = []
    images = []
    for entry in os.scandir(path):
        if entry.name.endswith('.png'):
            files.append(entry.path)
            image = Image.open(entry.path)
            images.append(array(image))
    return files, images


sans_serif_files, sans_serif_images = get_data(current_wd + '/' + 'sansserif')
sslabels = ['sansSerif' for i in range(len(sans_serif_images))]
np.random.shuffle(sans_serif_images)
serif_files, serif_images = get_data(current_wd + '/' + 'serif')
np.random.shuffle(serif_images)
slabels = ['serif' for i in range(len(serif_images))]
features = sans_serif_images + serif_images
labels = sslabels + slabels

# shuffling and splitting the data into training,validation and testing sets
training_sans_serif, eval_sans_serif = sans_serif_images[:-
                                                         num_eval_examples], sans_serif_images[-num_eval_examples:]
training_serif, eval_serif = serif_images[:-
                                          num_eval_examples], serif_images[-num_eval_examples:]
eval_sans_serif, test_sans_serif = eval_sans_serif[:- \
    num_test_examples], eval_sans_serif[-num_test_examples:]
eval_serif, test_serif = eval_serif[:- \
    num_test_examples], eval_serif[-num_test_examples:]


# saving the images in the training, evaluation and test sets in their
# corresponding folder
count = 0
for array in training_sans_serif:
    count += 1
    img = Image.fromarray(array)
    img.save("training_set/sansserif/image" + str(count) + ".png")

count = 0
for array in training_serif:
    count += 1
    img = Image.fromarray(array)
    img.save("training_set/serif/image" + str(count) + ".png")

count = 0
for array in eval_sans_serif:
    count += 1
    img = Image.fromarray(array)
    img.save("eval_set/sansserif/image" + str(count) + ".png")

count = 0
for array in eval_serif:
    count += 1
    img = Image.fromarray(array)
    img.save("eval_set/serif/image" + str(count) + ".png")

count = 0
for array in test_sans_serif:
    count += 1
    img = Image.fromarray(array)
    img.save("test_set/sansserif/image" + str(count) + ".png")

count = 0
for array in test_serif:
    count += 1
    img = Image.fromarray(array)
    img.save("test_set/serif/image" + str(count) + ".png")
