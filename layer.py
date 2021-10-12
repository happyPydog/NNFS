import numpy as np


#########
# layer #
#########

# Dense layer + L1 + L2
class Dense:
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
    def forward(self, inputs, training):
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

    # Retrieve layer parameters
    def get_parameters(self):
        return self.weights, self.biases

    # 接收參數
    def set_parameters(self, weights, biases):
        self.weights = weights
        self.biases = biases

# Dropout layer
class Dropout:

    # Init
    def __init__(self, rate):
        """Dropout rate 
        Dropout rate 是要丟掉的比例，
        e.g.,rate = 0.1，也就是保留 0.9 的 neuron.
        """
        self.rate = 1 - rate

    # forward pass
    def forward(self, inputs, training):
        # 保留 input values
        self.inputs = inputs
        # If not in the training mode - return values
        if not training:
            self.output = inputs.copy()
            # return None
            return
            
        # mask
        self.binary_mask = np.random.binomial(1, self.rate, 
                           size=inputs.shape) / self.rate
        # inputs * mask
        self.output = inputs * self.binary_mask

    # backward pass
    def backward(self, dvalues):
        # gradient on values
        self.dinputs = dvalues * self.binary_mask
        
# Input "layer"
class Layer_input:
    
    # forward pass
    def forward(self, inputs, training):
        self.output = inputs


#######################
# Activation function #
#######################

# ReLU
class ReLU:
    
    # forward pass
    def forward(self, inputs, training):
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
class Softmax:
    
    # forward pass
    def forward(self, inputs, training):
        
        # 取 np.exp 並且減最大值
        exp_values = np.exp(inputs - np.max(inputs, axis=1, 
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
    def forward(self, inputs, training):
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
    def forward(self, inputs, training):
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

#################
# Loss function #
#################

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

        # 累積loss
        self.accumulated_sum += np.sum(sample_losses)
        self.accumulated_count += len(sample_losses)

        # 只算 data loss (testing or validation 使用)
        if not include_regularization:
            return data_loss
        
        return data_loss, self.regularization_loss()

    # 計算累積loss
    def calculate_accumulated(self, *, include_regularization=False):

        # 計算平均loss
        data_loss = self.accumulated_sum / self.accumulated_count

        # If just data loss - return it
        if not include_regularization:
            return data_loss

        # Return the data and regularization losses
        return data_loss, self.regularization_loss()
    
    # reset sum 和 count
    def new_pass(self):
        self.accumulated_sum = 0
        self.accumulated_count = 0


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

    # 因為我們將 softmax + CrossEntropyLoss 當成特別的 case
    # 因此不需要使用 __init__ 和 forward
    # 它們都已經在 model 裡面被考慮過了

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