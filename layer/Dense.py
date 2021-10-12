import numpy as np
import nnfs
from nnfs.datasets import spiral_data


nnfs.init() # 如果需要重現性可以讀這行

# Dense layer
class Dense():
    # initialization
    def __init__(self, n_inpiuts, n_neurons):     
        # weights: 從 normal dist. sample 後乘上 0.01
        self.weights = 0.01 * np.random.randn(n_inpiuts, n_neurons) # 每個 neuron 都配一排 n_inputs 
        # bias
        self.biases = np.zeros((1, n_neurons)) # 每一個 neuron 都配一個 bias
    
    # forward pass
    def forward(self, inputs):
        # 把 inputs 存起來, backward 會用到
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases 
    
    # backward pass
    def backward(self, dvalues):
        """
        根據 dvalues,我們可以知道怎麼調 weights 才會降低 loss
        """
        # Gradient on parameters
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        # Gradient on values
        self.dinputs = np.dot(dvalues, self.weights.T)
    
if __name__ == "__main__":
    
    # 產生 dataset
    X, y = spiral_data(samples = 100, classes=3)
    
    # 產生 dense, ( input:2 -> ouput:3 )
    dense1 = Dense(2, 3)
    
    # forward
    dense1.forward(X)
    
    # 看 output 
    print(dense1.output[:5])