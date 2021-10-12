import numpy as np
import nnfs
from nnfs.datasets import spiral_data
from Dense import Dense
from Activation import ReLU, Softmax
from Loss import Softmax_CrossEntropyLoss
from Optimizer import SGD, AdaGrad, RMSProp, Adam

nnfs.init()

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
    optimizer = Adam(learning_rate=0.05, decay=5e-7)
    
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