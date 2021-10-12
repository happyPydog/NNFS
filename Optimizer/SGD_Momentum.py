import numpy as np
import nnfs
from nnfs.datasets import spiral_data
from Dense import Dense
from Activation import ReLU, Softmax
from Loss import Softmax_CrossEntropyLoss

nnfs.init()

# SGD optimizer
class SGD():
    
    # Initialize optimizer - set settings,
    # learning rate = 0.001 is default for this optimizer
    def __init__(self, learning_rate=1., decay=0., momentum=0.):
    	# initial learning rate
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.momentum = momentum
    
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

        # 如果有用 momentum
        if self.momentum:

            # 如果 layer 目前沒有 momentum 就產生
            # 初始值為 0 的 array
            if not hasattr(layer, 'weight_momentums'):
                layer.weight_momentums = np.zeros_like(layer.weights)
                layer.bias_momentums = np.zeros_like(layer.biases)

            # momentum weights
            weight_updatas = \
                self.momentum * layer.weight_momentums - \
                self.current_learning_rate * layer.dweights
            layer.weight_momentums = weight_updatas

            # momentum biases
            bias_updates = \
                 self.momentum * layer.bias_momentums - \
                 self.current_learning_rate * layer.dbiases
            layer.bias_momentums = bias_updates

        # Vanilla SGD updates (as before momentum update)
        else:
            weight_updatas = -self.current_learning_rate * \
                              layer.dweights
            bias_updates = -self.current_learning_rate * \
                            layer.dbiases

        # update wieghts and biases using either vanilla or momentum updates
        layer.weights += weight_updatas
        layer.biases += bias_updates

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
    optimizer = SGD(learning_rate=1., decay=1e-3, momentum=0.9)
    
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