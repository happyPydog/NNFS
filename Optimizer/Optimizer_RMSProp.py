import numpy as np
import nnfs
from nnfs.datasets import spiral_data
from Dense import Dense
from Activation import ReLU, Softmax
from Loss import Softmax_CrossEntropyLoss

nnfs.init()

class RMSProp():
    
    # Initialize optimizer - set settings,
    # learning rate = 0.001 is default for this optimizer
    def __init__(self, learning_rate=0.001, decay=0., epsilon=1e-7, rho=0.9):
    	# initial learning rate
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon
        self.rho = rho
    
    # 每一次更新參數前我們都做一次 decaying
    def pre_update_params(self):
    	"""Learning rate decay
    	隨著更新的 step 改變 learning rate,
    	此實作使用 「 exponential decay ］
    	"""
    	if self.decay:
    		self.current_learning_rate = self.learning_rate * \
    			(1. / (1 + self.decay * self.iterations))

    # update parameters
    def update_params(self, layer):

        # 如果 layer 目前沒有 cache 就產生
        # 初始值為 0 的 array
        if not hasattr(layer, 'weight_cache'):
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.biases)

        # 用 gradients 平方來更新 cache 
        # 累積過去所有的 gradient 平方和
        layer.weight_cache = self.rho * layer.weight_cache + \
                            (1 - self.rho) * layer.dweights**2
        layer.bias_cache = self.rho * layer.bias_cache + \
                            (1 - self.rho) * layer.dbiases**2

        # Vanilla SGD parameter update + normalization
        # with square rooted cache
        layer.weights += -self.current_learning_rate * \
                          layer.dweights / \
                          (np.sqrt(layer.weight_cache) + self.epsilon)
        layer.biases += -self.current_learning_rate * \
                        layer.dbiases / \
                        (np.sqrt(layer.bias_cache) + self.epsilon)

    def post_update_params(self):
    	self.iterations += 1


if __name__ == "__main__":
  
    # 產生 dataset
    X, y = spiral_data(samples=100, classes=3)
    
    # dense 1
    dense1 = Dense(2, 64)
    
    # relu
    activation1 = ReLU()
    
    # dense 2
    dense2 = Dense(64, 3)
    
    # softmax
    activation2 = Softmax()
    
    # loss function
    criterion = Softmax_CrossEntropyLoss()
    
    # create optimizer
    optimizer = RMSProp(learning_rate=0.02, decay=1e-5, rho=0.999)
    
    # set epoch
    epochs = 10001
    
    ################
    # Train in loop
    ################
    
    for epoch in range(epochs):
    
        ################
        # Forward pass
        ################
        dense1.forward(X)
        
        # dense1 -> relu
        activation1.forward(dense1.output)
        
        # relu -> dense2
        dense2.forward(activation1.output)
        
        # dense2 -> soft + Cross-entropy
        loss = criterion.forward(dense2.output, y)
        
        # accuracy
        predictions = np.argmax(criterion.output, axis=1)
        if len(y.shape) == 2:
            y = np.argmax(y, axis=1)
        # 取 mean 就等同於轉換成 %
        accuracy = np.mean(predictions == y)
        
        # 每100筆 data 就 print 出來看一下 accuracy
        if (epoch % 100) == 0:
            print(f'epoch: {epoch}, ' +
                  f'acc: {accuracy:.3f}, '+
                  f'loss: {loss:.3f},' +
                  f'lr: {optimizer.current_learning_rate}')
                  
        ################
        # Backward pass
        ################
        criterion.backward(criterion.output, y)
        dense2.backward(criterion.dinputs)
        activation1.backward(dense2.dinputs)
        dense1.backward(activation1.dinputs)
        
        # 更新參數
        optimizer.pre_update_params()
        optimizer.update_params(dense1)
        optimizer.update_params(dense2)
        optimizer.post_update_params()