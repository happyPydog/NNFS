import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import nnfs
import pickle
from nnfs.datasets import spiral_data, sine_data
from tqdm import tqdm

from layer import Dense, Layer_input, Dropout
from layer import ReLU, Softmax, Sigmoid, Linear
from layer import Softmax_CrossEntropyLoss, BinaryCrossEntropy, MSELoss, CrossEntropyLoss
from Optimizer import SGD, AdaGrad, RMSProp, Adam
from Accuracy import Accuracy_Regression, Accuracy_Categorical
from Model import Model

nnfs.init()

# Load a MNIST dataset
def load_mnist_dataset(dataset, path):

	# Scan all the directories and create a list of labels
	labels = os.listdir(os.path.join(path, dataset))
	# Create lists for samples and labels
	X = []
	y = []
	# For each label folder
	for label in tqdm(labels):
		for file in os.listdir(os.path.join(path, dataset, label)):
			image = cv2.imread(os.path.join(path, dataset, label, file), 
													cv2.IMREAD_UNCHANGED)
			# print(os.path.join(path, dataset, label, file))
			X.append(image)
			# print(image.shape)
			y.append(label)
	 # Convert the data to proper numpy arrays and return
	return np.array(X), np.array(y).astype('uint8')

# MNIST dataset (train + test)
def creat_data_mnist(path):

	# 產生 dataset
	X, y = load_mnist_dataset('train', path)
	X_test, y_test = load_mnist_dataset('test', path)

	return X, y, X_test, y_test	


if __name__ == '__main__':

	# Crete dataset
	X, y, X_test, y_test = creat_data_mnist('fashion_mnist_images')
	
	# Shuffle 
	keys = np.array(range(X.shape[0]))
	np.random.shuffle(keys)
	X = X[keys]
	y = y[keys]

	# Scale and reshape samples
	X = (X.reshape(X.shape[0], -1).astype(np.float32) - 127.5) / 127.5
	X_test = (X_test.reshape(X_test.shape[0], -1).astype(np.float32) - 127.5) / 127.5

	# Instantiate the model
	model = Model()

	# Add layers
	model.add(Dense(X.shape[1], 128))
	model.add(ReLU())
	model.add(Dense(128, 128))
	model.add(ReLU())
	model.add(Dense(128, 10))
	model.add(Softmax())

	# Set loss, optimizer and accuracy objects
	model.set(
	 loss=CrossEntropyLoss(),
	 optimizer=Adam(decay=5e-5),
	 accuracy=Accuracy_Categorical()
	)
	# Finalize the model
	model.finalize()
	# Train the model
	model.train(X, y, validation_data=(X_test, y_test),
	            epochs=10, batch_size=128, print_every=100)

	# Evaluate the model
	model.evaluate(X_test, y_test)

	# save model
	model.save('fashion_mnist_model')

	# load model
	model = Model.load('fashion_mnist_model')

	# Evaluate the model
	model.evaluate(X_test, y_test)


