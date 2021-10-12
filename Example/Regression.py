import numpy as np
import matplotlib.pyplot as plt
import nnfs
from nnfs.datasets import sine_data

from Activation import ReLU, Softmax, Sigmoid, Linear
from Loss import Softmax_CrossEntropyLoss, BinaryCrossEntropy, MSELoss
from Optimizer import SGD, AdaGrad, RMSProp, Adam
from Regularization import Dense

nnfs.init()

if __name__ == "__main__":

	# 產生 dataset
    X, y = sine_data()

    # dense 1
    dense1 = Dense(1, 64)
    
    # relu
    activation1 = ReLU()

    # dense 2
    dense2 = Dense(64, 64)

    # relu
    activation2 = ReLU()

    # dense 3
    dense3 = Dense(64, 1)

    # sigmiod
    activation3 = Linear()
    
    # loss function
    criterion = MSELoss()
    
    # create optimizer
    optimizer = Adam(learning_rate=0.005, decay=1e-3)

    # Accuracy precision for accuracy calculation
    # regression 的問題是沒有辦法計算真正的 accuracy
    # 但是我們可以算 simulate/approximate
    # how many values have a difference to their ground truth equivalent
    # less than given precision
    # We'll calculate this precision as a fraction of standard deviation
    # of al the ground truth values
    accuracy_precision = np.std(y) / 250

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

        # dense2 -> relu
        activation2.forward(dense2.output)

        # relu -> dense3
        dense3.forward(activation2.output)

        # dense3 -> Linear
        activation3.forward(dense3.output)

        # dense2 -> soft + Cross-entropy
        data_loss = criterion.calculate(activation3.output, y)

        # regularization penalty
        regularization_loss = \
            criterion.regularization_loss(dense1) + \
            criterion.regularization_loss(dense2) + \
            criterion.regularization_loss(dense3)

        # 計算所有的 loss 
        loss = data_loss + regularization_loss

        # Calculate accuracy from output of activation2 and targets    
        # Part in the brackets returns a binary mask - array consisting    
        # of True/False values, multiplying it by 1 changes it into array    
        # of 1s and 0s    
        predictions = activation3.output
        accuracy = np.mean(np.absolute(predictions - y) < 
        				   accuracy_precision)
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
        criterion.backward(activation3.output, y)
        activation3.backward(criterion.dinputs)
        dense3.backward(activation3.dinputs)
        activation2.backward(dense3.dinputs)
        dense2.backward(activation2.dinputs)
        activation1.backward(dense2.dinputs)
        dense1.backward(activation1.dinputs)

        # 更新參數
        optimizer.pre_update_params()
        optimizer.update_params(dense1)
        optimizer.update_params(dense2)
        optimizer.update_params(dense3)
        optimizer.post_update_params()


    # drawing output with testing data
    X_test, y_test = sine_data()
    dense1.forward(X_test)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    activation2.forward(dense2.output)
    dense3.forward(activation2.output)
    activation3.forward(dense3.output)

    plt.plot(X_test, y_test)
    plt.plot(X_test, activation3.output)
    plt.show()