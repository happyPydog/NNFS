import numpy as np

# Accuracy calculation for regression model
class Accuracy:

	def calculate(self, predictions, y):

		# 得到比較的結果
		comparisons = self.compare(predictions, y)
		# 計算 acc
		accuracy = np.mean(comparisons)

		# Add accumulated sum of matching values and sample count
		self.accumulated_sum += np.sum(comparisons)
		self.accumulated_count += len(comparisons) 

		return accuracy

	# Calculates accumulated accuracy
	def calculate_accumulated(self):

		# Calculate an accuracy
		accuracy = self.accumulated_sum / self.accumulated_count

		# Return the data and regularization losses
		return accuracy
		
	# Reset variables for accumulated accuracy
	def new_pass(self):
		self.accumulated_sum = 0
		self.accumulated_count = 0

# Accuracy calculation for classification model
class Accuracy_Categorical(Accuracy):

	def __init__(self, *, binary=False):
		# Binary mode?
		self.binary = binary

	# 不需要 initialization
	def init(self, y):
		pass

	# compare
	def compare(self, predictions, y):
		if not self.binary and len(y.shape) == 2:
			y = np.argmax(y, axis=1)

		return predictions == y
		
# Regression model Accuracy
class Accuracy_Regression(Accuracy):

	def __init__(self):
		# 產生 precision property
		self.precision = None

	# 計算 precision value based on passed-in ground truth
	def init(self, y, reinit=False):
		if self.precision is None or reinit:
			self.precision = np.std(y) / 250

	# 預測值跟真實值之間去衡量差異or距離
	def compare(self, predictions, y):
		return np.absolute(predictions - y) < self.precision