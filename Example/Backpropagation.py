import numpy as np
import nnfs
from nnfs.datasets import spiral_data
from Dense import Dense
from Activation import ReLU, Softmax
from Loss import Softmax_CrossEntropyLoss

nnfs.init()

if __name__ == "__main__":
    
    # 產生 dataset
    X, y = spiral_data(samples=100, classes=3)
    
    # dense 1
    dense1 = Dense(2, 3)
    
    # relu
    activation1 = ReLU()
    
    # dense 2
    dense2 = Dense(3, 3)
    
    # loss function
    criterion = Softmax_CrossEntropyLoss()
    
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
    
    # 看一下前五筆資料的損失
    print('output:')
    print(criterion.output[:5])
    
    # print loss value
    print('loss:', loss)
    
    # accuracy
    predictions = np.argmax(criterion.output, axis=1)
    if len(y.shape) == 2:
        y = np.argmax(y, axis=1)
    # 取 mean 就等同於轉換成 %
    accuracy = np.mean(predictions == y)
    
    # 看一下 accuracty 是多少
    print('acc:', accuracy)
    
    ################
    # Backward pass
    ################
    criterion.backward(criterion.output, y)
    dense2.backward(criterion.dinputs)
    activation1.backward(dense2.dinputs)
    dense1.backward(activation1.dinputs)
    
    # print 出每一層的 dweights & biases
    # dweights 就是該 weights 對 loss 的影響力
    print('每一層的參數:')
    print(dense1.dweights)
    print(dense1.dbiases)
    print(dense2.dweights)
    print(dense2.dbiases)