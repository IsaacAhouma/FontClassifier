In this programming challenge, we build a classifier in TensorFlow to distinguish between serif and sans serif letters found in a data folder.

My solution to the programming challenge is split into 3 scripts:
	1)preprocessing.py in which data processing tasks are performed and the data is split into training, evaluation and testing sets.
	2)cnn_classifier.py in which the actual deep learning model is built and trained.
	3)make_predictions.py which shows how the model can be loaded and used to make predictions on new data.

My trained model is saved under 'fontclassifier.h5' and can be directly loaded to make predictions on new data. Examples of how to do this are given in make_predictions.py


Note:
TensorFlow is not supported by some releases of python 3.6, so we might need to run the project using python 3.5.
To create and activate a virtual python 3.5 environment please download python 3.5 (if not already downloaded) and run the following from the command line:
conda create -n python35 python=3.5
source activate python35

To download all required packages please run 'pip install -r requirements.txt' from the working directory where this folder is saved.