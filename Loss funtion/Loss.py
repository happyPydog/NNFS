import numpy as np
import nnfs
from nnfs.datasets import spiral_data
from Dense import Dense
from Activation import ReLU, Softmax

nnfs.init()

# 基本的loss, 給其他loss繼承
class Loss():
    """計算正規的損失
    給定 model 的 output , 
    去計算它跟真實值之間的差距。
    """
    # Set/remember trainable layers
    def remember_trainable_layers(self, trainable_layers):
        self.trainable_layers = trainable_layers

    # Calculates the data and regularization losses
    # given model output and ground truth values
    def calculate(self, output, y, *, include_regularization=False):
        
        # 計算 sample losses
        sample_losses = self.forward(output, y)
        
        # 計算平均損失
        data_loss = np.mean(sample_losses)

        # 只算 data loss (testing or validation 使用)
        if not include_regularization:
            return data_loss
        
        return data_loss, self.regularization_loss()

    # regularization
    def regularization_loss(self):

        # 0 by default
        regularization_loss = 0

        # Calculate regularization loss
        # iterate all trainable layers
        for layer in self.trainable_layers:

            # L1 regularization - weights
            if layer.weight_regularizer_l1 > 0:
                regularization_loss += layer.weight_regularizer_l1 * \
                                       np.sum(np.abs(layer.weights))

            # L2 regularization - weights
            if layer.weight_regularizer_l2 > 0:
                regularization_loss += layer.weight_regularizer_l2 * \
                                       np.sum(layer.weights * layer.weights)
            
            # L1 regularization - biases
            if layer.bias_regularizer_l1 > 0:
                regularization_loss += layer.bias_regularizer_l1 * \
                                       np.sum(np.abs(layer.biases))

            # L2 regularization - biases
            if layer.bias_regularizer_l2 > 0:
                regularization_loss += layer.bias_regularizer_l2 * \
                                       np.sum(layer.biases * layer.biases)

        return regularization_loss
            
# cross-entropy loss
class CrossEntropyLoss(Loss):
    """
    計算類別型的損失
    """
    # forward pass
    def forward(self, y_pred, y_true):
        
        # 樣本數
        batch_size = len(y_pred)
        
        # clip data 以防除以 0
        # 兩邊都要 clip 以防 mean 改變
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7) # 1e-7 是很小的數值,可更換
        
        # 依照有沒有 one-hot encoded 的 target 去計算  loss
        # for categorical labels
        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(batch_size), y_true]
            
        # for one-hot encoded labels
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped*y_true, axis=1)
            
        # -log
        negative_log_likelihoods = -np.log(correct_confidences)
        
        return negative_log_likelihoods
    
    # backward pass
    def backward(self, dvalues, y_true):
        
        # 樣本數
        batch_size = len(dvalues)
        # Number of samples
        # 我們會用第一個 sample 來計算
        labels = len(dvalues[0])
        
        # 如果 labels 是 sparse, 則將它們轉成 one-hot vector
        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]
        
        # 計算 gradient
        self.dinputs = -y_true / dvalues
        
        # Normalize gradient
        self.dinputs = self.dinputs / batch_size

# Softmax classifier - combined Softmax activations
# and Cross-entropy loss for faster backward step
class Softmax_CrossEntropyLoss():
    """結合 Softmax + CrossEntropy
    Softmax + Crossentropy 是最常使用的 output layer activation 和
    criterion, 因此我們直接把兩者結合放在一起.
    """
    def __init__(self):
        self.activation = Softmax()
        self.loss = CrossEntropyLoss()
        
    # forward pass
    def forward(self, inputs, y_true):
        
        self.activation.forward(inputs) # output -> softmax
        self.output = self.activation.output
        
        return self.loss.calculate(self.output, y_true)
    
    # backward pass
    def backward(self, dvalues, y_true):
        
        # 樣本數
        batch_size = len(dvalues)
        
        # 如果是 one-hot encoded 就轉成 discrete values
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)
        
        # copy so we can safely modify
        self.dinputs = dvalues.copy()
        # 計算 gradient
        self.dinputs[range(batch_size), y_true] -= 1
        # Normalize
        self.dinputs = self.dinputs / batch_size

# Binary Cross-Entropy Loss
class BinaryCrossEntropy(Loss):

    # forward pass
    def forward(self, y_pred, y_true):

        # clip 防止除以 0
        # clip both sides 保持 mean 不變
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        # sample loss
        sample_losses = -(y_true * np.log(y_pred_clipped) +
                          (1 - y_true) * np.log(1 - y_pred_clipped))
        # 對整筆 sample 取平均
        # axis = -1 是為了取最後一個維度來平均，也就是一筆資料的 output 取平均
        sample_losses = np.mean(sample_losses, axis=-1)

        return sample_losses

    # backward pass
    def backward(self, dvalues, y_true):

        # 樣本數
        samples = len(dvalues)
        # 每一筆樣本的 output 數量
        outputs = len(dvalues[0])

        # clip
        clipped_dvalues = np.clip(dvalues, 1e-7, 1 - 1e-7)

        # 計算 gradient
        self.dinputs = -(y_true / clipped_dvalues -
                         (1 - y_true) / (1 - clipped_dvalues)) / outputs

        # Normalize gradient
        self.dinputs = self.dinputs / samples

# Mean Squared Error loss
class MSELoss(Loss): # L2 loss
    
    # forward pass
    def forward(self, y_pred, y_true):

        # 計算 loss
        sample_losses = np.mean((y_true - y_pred)**2, axis=-1)

        return sample_losses

    # backward pass
    def backward(self, dvalues, y_true):

        # 樣本數
        samples = len(dvalues)
        # 每一筆樣本的 output 數量
        outputs = len(dvalues[0])

        # gradient
        self.dinputs = -2 * (y_true - dvalues) / outputs
        # Normalize gradient
        self.dinputs = self.dinputs / samples

# Mean Absolute Error loss
class L1Loss(Loss):

    # forward pass
    def forward(slef, y_pred, y_true):

        # 計算 loss
        sample_losses = np.mean(np.abs(y_true - y_pred), axis=-1)

        return sample_losses

    # backward pass
    def backward(self, dvalues, y_true):

        # 樣本數
        samples = len(dvalues)
        # 每一筆樣本的 output 數量
        outputs = len(dvalues[0])

        # gradient
        # The sign function returns:
        # -1 if x < 0, 0 if x==0, 1 if x > 0.
        self.dinputs = np.sign(y_true - dvalues) / outputs
        # Normalize gradient
        self.dinputs = self.dinputs / samples


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
    
    # loss function
    criterion = CrossEntropyLoss()
    
    # loss
    loss = criterion.calculate(activation2.output, y)
    
    # 看一下這筆資料的平均損失
    print("loss:", loss)