import numpy as np
import nnfs
from nnfs.datasets import spiral_data
from Activation import ReLU, Softmax, Sigmoid
from Loss import Softmax_CrossEntropyLoss, BinaryCrossEntropy
from Optimizer import SGD, AdaGrad, RMSProp, Adam
from Regularization import Dense

nnfs.init()

if __name__ == "__main__":
  
    # 產生 dataset
    X, y = spiral_data(samples=100, classes=2)

    # 改變 y 的 shape
    y = y.reshape(-1, 1) # (sample_size, 1)

    # dense 1
    dense1 = Dense(2, 64, weight_regularizer_l2=5e-4,
                          bias_regularizer_l2=5e-4)
    
    # relu
    activation1 = ReLU()

    # dense 2
    dense2 = Dense(64, 1)

    # sigmiod
    activation2 = Sigmoid()
    
    # loss function
    criterion = BinaryCrossEntropy()
    
    # create optimizer
    optimizer = Adam(decay=5e-7)
    
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

        # dense2 -> sigmiod
        activation2.forward(dense2.output)

        # dense2 -> soft + Cross-entropy
        data_loss = criterion.calculate(activation2.output, y)

        # regularization penalty
        regularization_loss = \
            criterion.regularization_loss(dense1) + \
            criterion.regularization_loss(dense2)

        # 計算所有的 loss 
        loss = data_loss + regularization_loss

        # Calculate accuracy from output of activation2 and targets    
        # Part in the brackets returns a binary mask - array consisting    
        # of True/False values, multiplying it by 1 changes it into array    
        # of 1s and 0s    
        predictions = (activation2.output > 0.5) * 1    
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
        criterion.backward(activation2.output, y)
        activation2.backward(criterion.dinputs)
        dense2.backward(activation2.dinputs)
        activation1.backward(dense2.dinputs)
        dense1.backward(activation1.dinputs)

        # 更新參數
        optimizer.pre_update_params()
        optimizer.update_params(dense1)
        optimizer.update_params(dense2)
        optimizer.post_update_params()

    # Validate the model
    # Create test dataset
    X_test, y_test = spiral_data(samples=100, classes=2)

    # change y_test's shape for BCE loss
    y_test = y_test.reshape(-1, 1)

    # Perform a forward pass of our testing data through this layer
    dense1.forward(X_test)

    # Perform a forward pass through activation function
    # takes the output of first dense layer here
    activation1.forward(dense1.output)

    # Perform a forward pass through second Dense layer
    # takes outputs of activation function of first layer as inputs
    activation2.forward(activation1.output)

    # Perform a forward pass through the activation/loss function
    # takes the output of second dense layer here and returns loss
    loss = criterion.calculate(activation2.output, y_test)

    # Calculate accuracy from output of activation2 and targets    
    # Part in the brackets returns a binary mask - array consisting    
    # of True/False values, multiplying it by 1 changes it into array    
    # of 1s and 0s    
    predictions = (activation2.output > 0.5) * 1    
    accuracy = np.mean(predictions == y_test)

    print(f'validation, acc: {accuracy:.3f}, loss: {loss:.3f}')