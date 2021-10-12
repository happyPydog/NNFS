import numpy as np
import nnfs
from nnfs.datasets import spiral_data
from Activation import ReLU, Softmax
from Loss import Softmax_CrossEntropyLoss
from Optimizer import SGD, AdaGrad, RMSProp, Adam

nnfs.init()

# Dense layer + L1 + L2
class Dense():
    # initialization
    def __init__(self, n_inpiuts, n_neurons,
                 weight_regularizer_l1=0, weight_regularizer_l2=0,
                 bias_regularizer_l1=0, bias_regularizer_l2=0):

        # weights: 從 normal dist. sample 後乘上 0.01
        self.weights = 0.01 * np.random.randn(n_inpiuts, n_neurons) # 每個 neuron 都配一排 n_inputs 
        # bias
        self.biases = np.zeros((1, n_neurons)) # 每一個 neuron 都配一個 bias

        # regularization
        self.weight_regularizer_l1 = weight_regularizer_l1
        self.weight_regularizer_l2 = weight_regularizer_l2
        self.bias_regularizer_l1 = bias_regularizer_l1
        self.bias_regularizer_l2 = bias_regularizer_l2

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

        # Gradients on regularization
        # L1 on weights
        if self.weight_regularizer_l1 > 0:
            dL1 = np.ones_like(self.weights)
            dL1[self.weights < 0] = -1
            self.dweights += self.weight_regularizer_l1 * dL1
        # L2 on weights
        if self.weight_regularizer_l2 > 0:
            self.dweights += 2 * self.weight_regularizer_l2 * self.weights

        # L1 on biases
        if self.bias_regularizer_l1 > 0:
            dL1 = np.ones_like(self.biases)
            dL1[self.biases < 0] = -1
            self.dbiases += self.bias_regularizer_l1 * dL1
        # L2 on biases
        if self.bias_regularizer_l2 > 0:
            self.dbiases += 2 * self.bias_regularizer_l2 * self.biases

        # Gradient on values
        self.dinputs = np.dot(dvalues, self.weights.T)
        
# Input "layer"
class Layer_input:
    
    # forward pass
    def forward(self, inputs):
        self.output = inputs

if __name__ == "__main__":
  
    # 產生 dataset
    X, y = spiral_data(samples=1000, classes=3)
    
    # dense 1
    dense1 = Dense(2, 512, weight_regularizer_l2=5e-4,
                          bias_regularizer_l2=5e-4)
    
    # relu
    activation1 = ReLU()
    
    # dense 2
    dense2 = Dense(512, 3)
    
    # loss function
    criterion = Softmax_CrossEntropyLoss()
    
    # create optimizer
    optimizer = Adam(learning_rate=0.02, decay=5e-7)
    
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
        data_loss = criterion.forward(dense2.output, y)

        # regularization penalty
        regularization_loss = \
            criterion.loss.regularization_loss(dense1) + \
            criterion.loss.regularization_loss(dense2)

        # 計算所有的 loss 
        loss = data_loss + regularization_loss

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
                  f'loss: {loss:.3f} (' +
                  f'data_loss: {data_loss:.3f}, ' +
                  f'reg_loss: {regularization_loss:.3f}), ' + 
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

    # Validate the model
    # Create test dataset
    X_test, y_test = spiral_data(samples=100, classes=3)
    # Perform a forward pass of our testing data through this layer
    dense1.forward(X_test)
    # Perform a forward pass through activation function
    # takes the output of first dense layer here
    activation1.forward(dense1.output)
    # Perform a forward pass through second Dense layer
    # takes outputs of activation function of first layer as inputs
    dense2.forward(activation1.output)
    # Perform a forward pass through the activation/loss function
    # takes the output of second dense layer here and returns loss
    loss = criterion.forward(dense2.output, y_test)
    # Calculate accuracy from output of activation2 and targets
    # calculate values along first axis
    predictions = np.argmax(criterion.output, axis=1)
    if len(y_test.shape) == 2:
        y_test = np.argmax(y_test, axis=1)
    accuracy = np.mean(predictions == y_test)
    print(f'validation, acc: {accuracy:.3f}, loss: {loss:.3f}')