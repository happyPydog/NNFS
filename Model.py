import numpy as np
import matplotlib.pyplot as plt
import nnfs
import pickle
import copy
from nnfs.datasets import spiral_data, sine_data
from tqdm import tqdm

from layer import Dense, Layer_input, Dropout
from layer import ReLU, Softmax, Sigmoid, Linear
from layer import Softmax_CrossEntropyLoss, BinaryCrossEntropy, MSELoss, CrossEntropyLoss
from Optimizer import SGD, AdaGrad, RMSProp, Adam
from Accuracy import Accuracy_Regression, Accuracy_Categorical

nnfs.init()


# Model class
class Model:

	def __init__(self):
		# 產生 list 來裝 network object
		self.layers = []
		# Soft,ax classifier's output object
		self.softmax_classifier_output = None

	def add(self, layer):
		self.layers.append(layer)

	# Set loss and optimizer
	# * 用來表示後面的參數都必須是 keyword arguments
	def set(self, *, loss=None, optimizer=None, accuracy=None):

		if loss is not None:
			self.loss = loss

		if optimizer is not None:
			self.optimizer = optimizer

		if accuracy is not None:
			self.accuracy = accuracy

	# Finalize the model
	def finalize(self):
		"""Finalize
		我們要紀錄 layer 的前一層和下一層的屬性，
		i.e. , layer.prev 和 layer.next，
		有了這些資訊我們可以用 for-loop 來完成 forward pass.
		"""
		# Create and set the input layer
		self.input_layer = Layer_input()

		# 計算有幾層 layer
		layer_count = len(self.layers)

		# Initialize a list containing trainable layers:
		self.trainable_layers = []

		# Interate the objects
		for i in range(layer_count):

			# 如果是第一層 
			# 那它的前一層就會是特殊的 Layer_input
			if i == 0:
				self.layers[i].prev = self.input_layer
				self.layers[i].next = self.layers[i+1]

			# hidden layer (except the first and the last)
			elif i < layer_count - 1:
				self.layers[i].prev = self.layers[i-1]
				self.layers[i].next = self.layers[i+1]

			# last layer - the next object is the loss
			else:
				self.layers[i].prev = self.layers[i-1]
				self.layers[i].next = self.loss
				# 紀錄我們最後一層使用的 activations
				self.output_layer_activation = self.layers[i]

			# 檢查 layer 是否有 "weights",
			# 如果有則代表這層是可以訓練的
			# 那麼就把此 layer 加入倒 trainable_layers
			if hasattr(self.layers[i], "weights"):
				self.trainable_layers.append(self.layers[i])

		# 更新 loss object with trainable layers
		if self.loss is not None:
			self.loss.remember_trainable_layers(self.trainable_layers)

		# 因為我們的 softmax 和 CrossEntropyLoss 是寫在一起的
		# 因此 forward 會有問題
		# 所以這裡要先判斷我們使用的 last layer 和 loss function
		# 是否為 softmax + CrossEntropyLoss
		# isinstance(): 判斷是否為某個已知的類型
		if isinstance(self.layers[-1], Softmax) and \
			isinstance(self.loss, CrossEntropyLoss):
			# Create an object of combined activation and loss functions
			self.softmax_classifier_output = Softmax_CrossEntropyLoss()

	# forward pass
	def forward(self, X, training):

		# input_layer 要先做 forward 才有 output 屬性
		# 這樣接下來的 layer.prev.output 才能從 i == 1 成功執行
		self.input_layer.forward(X, training)

		# Call forward method of every object in a chain
		# Pass output of the previous object as a parameter
		for layer in self.layers:
			layer.forward(layer.prev.output, training)

		# "layer" is now the last object from the list,
		# return its output
		return layer.output

	# backward pass
	def backward(self, output, y):

		# 如果使用 Softmax classifier
		if self.softmax_classifier_output is not None:
			# call backward method 
			self.softmax_classifier_output.backward(output, y)

			# 因為 last layer 的 backward 已經被算過了
			# i.e. , softmax_classifier_output.dinputs
			# 因此不需要做 backward
			self.layers[-1].dinputs = self.softmax_classifier_output.dinputs

			# 接下來要做的事情跟原本一樣
			# 只是扣掉最後一層
			for layer in reversed(self.layers[:-1]):
				layer.backward(layer.next.dinputs)

			# return None
			return

		# 從 loss function 計算出第一層 backward 的 dinputs
		self.loss.backward(output, y)

		# 接下來每一層都做 for-loop 算出 dinputs
		for layer in reversed(self.layers):
			layer.backward(layer.next.dinputs)
	# train
	def train(self, X, y, *, epochs=1, batch_size=None,
			  print_every=1, validation_data=None):

		# Initialize accuracy object
		self.accuracy.init(y)

		# 如果不使用 batch_size
		train_steps = 1

		# 如果有用 validation data 就設定 validation steps
		if validation_data is not None:
			validation_steps = 1

			X_val, y_val = validation_data

		# number of steps
		if batch_size is not None:
			train_steps = len(X) // batch_size

			if train_steps * batch_size < len(X):
				train_steps += 1

			if validation_data is not None:
				validation_steps = len(X_val) // batch_size

				if  validation_steps * batch_size < len(X_val):
					validation_steps += 1

		# Main training loop
		for epoch in range(1, epochs+1):

			# print epoch number
			print(f'epoch: {epoch}')

			# Reset accumulated values
			self.loss.new_pass()
			self.accuracy.new_pass()

			for step in range(train_steps):

				# if batch_size is not set
				# train using one step and full dataset
				if batch_size is None:
					batch_X = X
					batch_y = y

				# Otherwise
				else:
					batch_X = X[step*batch_size:(step+1)*batch_size]
					batch_y = y[step*batch_size:(step+1)*batch_size]

				##############
				# forward pass
				##############
				output = self.forward(X, training=True)

				# 計算 loss
				data_loss, regularization_loss = \
					self.loss.calculate(output, y, 
										include_regularization=True)
				loss = data_loss + regularization_loss

				# predicaitons and calculate accuracy
				predictions = self.output_layer_activation.predictions(
								  output)
				accuracy = self.accuracy.calculate(predictions, y)

				###############
				# backward pass
				###############
				self.backward(output, y)

				# update parameters
				self.optimizer.pre_update_params()
				for layer in self.trainable_layers:
					self.optimizer.update_params(layer)
				self.optimizer.post_update_params()

				# Print a summary
				if not step % print_every or step == train_steps - 1:
					print(f'step: {step}, ' +
	                  	  f'acc: {accuracy:.3f}, '+
	                  	  f'loss: {loss:.3f} (' +
	                  	  f'data_loss: {data_loss:.3f}, ' +
	                  	  f'reg_loss: {regularization_loss:.3f}), ' + 
	                  	  f'lr: {self.optimizer.current_learning_rate}')

			# Get and print epoch loss and accuracy
			epoch_data_loss, epoch_regularization_loss = \
			self.loss.calculate_accumulated(
			include_regularization=True)
			epoch_loss = epoch_data_loss + epoch_regularization_loss
			epoch_accuracy = self.accuracy.calculate_accumulated()
			print(f'training, ' +
			 	  f'acc: {epoch_accuracy:.3f}, ' +
			 	  f'loss: {epoch_loss:.3f} (' +
			 	  f'data_loss: {epoch_data_loss:.3f}, ' +
			 	  f'reg_loss: {epoch_regularization_loss:.3f}), ' +
			 	  f'lr: {self.optimizer.current_learning_rate}')

		# 如果有使用 validation data 
		if validation_data is not None:
			# Evaluate the model
			self.evaluate(*validation_data, batch_size=batch_size)

	# evaluation
	def evaluate(self, X_val, y_val, *, batch_size=None):

		validation_steps = 1

		if batch_size is not None:
			validation_steps = len(X_val) // batch_size
			if validation_steps * batch_size < len(X_val):
				validation_steps += 1

		self.loss.new_pass()
		self.accuracy.new_pass()

		# Iterate over steps
		for step in range(validation_steps):
			if batch_size is None:
				batch_X = X_val
				batch_y = y_val
			else:
				batch_X = X_val[step*batch_size:(step+1)*batch_size]
				batch_y = y_val[step*batch_size:(step+1)*batch_size]

			# forward pass
			output = self.forward(X_val, training=False)

			# Calculate the loss
			self.loss.calculate(output, batch_y)

			# predictions & accuracy
			predictions = self.output_layer_activation.predictions(
							  output)
			self.accuracy.calculate(predictions, batch_y)

		# Get and print validation loss and accuracy
		validation_loss = self.loss.calculate_accumulated()
		validation_accuracy = self.accuracy.calculate_accumulated()

		# Print a summary
		print(f'validation, ' +
			  f'acc: {validation_accuracy:.3f}, ' +
			  f'loss: {validation_loss:.3f}')

	# Retrieves and returns parameters of trainable layers
	def get_parameters(self):

		parameters = []

		for layer in self.trainable_layers:
			parameters.append(layer.get_parameters())

		return parameters

	# 接收參數
	def set_parameters(self, parameters):

		for parameter_set, layer in zip(parameters, self.trainable_layers):
			layer.set_parameters(*parameter_set)

	# 儲存 model parameters
	def save_parameters(self, path):
		with open(path, 'wb') as f:
			pickle.dump(self.get_parameters(), f)

	# load pre-trained weights
	def load_parameters(self, path):
		with open(path, 'rb') as f:
			self.set_parameterst(pickle.load(f))

	# Save model
	def save(self, path):

		# Make a deep copy of current model instance
		model = copy.deepcopy(self)
		# Reset accumulated values in loss and accuracy objects
		model.loss.new_pass()
		model.accuracy.new_pass()
		# Remove data from the input layer
		# and gradients from the loss object
		model.input_layer.__dict__.pop('output', None)
		model.loss.__dict__.pop('dinputs', None)
		# For each layer remove inputs, output and dinputs properties
		for layer in model.layers:
			for property in ['inputs', 'output', 'dinputs', 'dweights', 'dbiases']:
				layer.__dict__.pop(property, None)
		# Open a file in the binary-write mode and save the model
		with open(path, 'wb') as f:
			pickle.dump(model, f)

	# Loads and returns a model
	@staticmethod
	def load(path):

		# Open file in the binary-read mode, load a model
		with open(path, 'rb') as f:
			model = pickle.load(f)
			
		# Return a model
		return model

	# Predict on new dataset
	def prediction(self, X, *, batch_size=None):
		# Default value if batch size is not being set
		prediction_steps = 1
		# Calculate number of steps
		if batch_size is not None:
			prediction_steps = len(X) // batch_size
			if prediction_steps * batch_size < len(X):
				prediction_steps += 1

		# Model outputs
		output = []
		# Iterate over steps
		for step in range(prediction_steps):
			if batch_size is None:
				batch_X = X
			else:
				batch_X = X[step*batch_size:(step+1)*batch_size]

		# Perform the forward pass
		batch_output = self.forward(batch_X, training=False)
		# Append batch prediction to the list of predictions
		output.append(batch_output)
		
		# Stack and return results
		return np.vstack(output)


if __name__ == "__main__":

	# Create train and test dataset
	X, y = spiral_data(samples=1000, classes=3)
	X_test, y_test = spiral_data(samples=100, classes=3)
	
	# Instantiate the model
	model = Model()

	# Add layers
	model.add(Dense(2, 512, weight_regularizer_l2=5e-4,
	                             bias_regularizer_l2=5e-4))
	model.add(ReLU())
	model.add(Dropout(0.1))
	model.add(Dense(512, 3))
	model.add(Softmax())

	# Set loss, optimizer and accuracy objects
	model.set(
	 loss=CrossEntropyLoss(),
	 optimizer=Adam(learning_rate=0.05, decay=5e-5),
	 accuracy=Accuracy_Categorical()
	)
	# Finalize the model
	model.finalize()
	# Train the model
	model.train(X, y, validation_data=(X_test, y_test),
	            epochs=10000, print_every=100)


