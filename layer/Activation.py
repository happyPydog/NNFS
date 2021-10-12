from Dense import Dense
import numpy as np
import nnfs
from nnfs.datasets import spiral_data

nnfs.init()

# ReLU
class ReLU():
    
    # forward pass
    def forward(self, inputs):
        # 紀錄 inputs
        self.inputs = inputs
        self.output = np.maximum(0, inputs)
    
    # backward pass
    def backward(self, dvalues):
        # dvalues 還會用到,因此用copy()來操作
        self.dinputs = dvalues.copy()
        
        # 如果當初的 inputs < 0 則回傳的 dvalues=0
        self.dinputs[self.inputs <= 0] = 0

    # 計算 outputs 的 predications
    def predictions(self, outputs):
        return outputs

# Sortmax
class Softmax():
    
    # forward pass
    def forward(self, inputs):
        
        # 取 np.exp 並且減最大值
        exp_values = np.exp(inputs - np.max(inputs, 
                                            axis=1, 
                                            keepdims=True))
        
        # Normalize
        prob = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        
        self.output = prob
    
    # backward pass
    def backward(self, dvalues):
        # 產生 dvaleus.shape 
        self.dinputs = np.empty_like(dvalues)
        # enumerate output and gradient
        for index, (sigle_output, sigle_dvalues) in \
                enumerate(zip(self.output, dvalues)):
                    # flatten output 
                    single_output = sigle_output.reshape(-1, 1)
                    # 計算 Jacobian matrix of the output
                    jacobian_matrix = np.diagflat(sigle_output) - \
                                      np.dot(sigle_output, sigle_value.T)
                    # 計算 sample-wise gradient
                    self.dinputs[index] = np.dot(jacobian_matrix, sigle_dvalues)

    # 計算 outputs 的 predications
    def predictions(self, outputs):
        return np.argmax(outputs, axis=1)

# Sigmoid
class Sigmoid:

    # forward pass
    def forward(self, inputs):
        self.inputs = inputs
        self.output = 1 / (1 + np.exp(-inputs))

    # backward pass
    def backward(self, dvalues):
        # Derivative - calculates from output of the sigmoid function
        self.dinputs = dvalues * (1 - self.output) * self.output

    # 計算 outputs 的 predications
    def predictions(self, outputs):
        # "0.5"是自己設定的 threshold
        return (outputs > 0.5) * 1 

# Linear
class Linear:
    """Linear function
    Linear -> inputs = output
    (其實也可以不使用 linear activation,
     使用的目的在於比較清楚的表達這層的 output
     用的是 lienar fucntion.)
    """
    # forward pass
    def forward(self, inputs):
        # Just remember values
        self.inputs = inputs
        self.output = inputs

    # backward pass
    def backward(self, dvalues):
        # derivative is 1, 1 * dvalues = dvalues - the chain rule
        self.dinputs = dvalues.copy()

    # 計算 outputs 的 predications
    def predictions(self, outputs):
        return outputs

if __name__ == "__main__":
    
    # 產生 dataset
    X, y = spiral_data(samples=100, classes=3)
    
    # dense 1
    dense1 = Dense(2, 3)
    
    # relu
    activation1 = ReLU()
    
    # dense 2
    dense2 = Dense(3, 3)
    
    # softmax
    activation2 = Softmax()
    
    # forward
    dense1.forward(X)
    
    # dense1 -> relu
    activation1.forward(dense1.output)
    
    # relu -> dense2
    dense2.forward(activation1.output)
    
    # dense2 -> softmax
    activation2.forward(dense2.output)
    
    # end
    
    # 看一下最後的 output
    print(activation2.output[:5])