'''import numpy as np

#input features (x1, x2)
X = np.array([1.0, 2.0])

#weights and bias
weights = np.array([0.4, 0.6])
bias = 0.5

#activation function (ReLU)
def relu(x):
    return np.maximum(0, x)

# neuron output
z = np.dot(weights, X) + bias 
output = relu(z)

print("Neuron output:", output)
# FNN Simple Feedforward Neural Network using NumPy
import numpy as np

# input features (2 sample, 2 features each)

X = np.array([[0, 1],
              [1,1]])

# weights for hidden layer (2 input  3 hidden each)
w_hidden = np.array([[0.2, 0.4, 0.6],
                     [0.5, 0.1, 0.3]])
b_hidden = np.array([0.1, 0.2, 0.3])

# weights for output layer (3 hidden 1 output neuron)
w_output = np.array([[0.3],
                     [0.7],
                     [0.5]])
b_output = np.array([0.1]) 

# activation function: ReLU
def relu(x):
    return np.maximum(0, x)

# forward pass 
hidden_input = np.dot(X, w_hidden) + b_hidden
hidden_output = relu(hidden_input)

output = np.dot(hidden_output, w_output) + b_output
print("Final Output:\n", output)'''

# One Neuron with Manual Backpropagation (Using NumPy)

import numpy as np

# input and expected output
X = np.array([[1.0],[2.0],[3.0]])
Y = np.array([[2.0],[4.0],[6.0]])

# Initialize weights and bias

w = np.random.rand(1)
b = np.random.rand(1)
lr = 0.01 # learning rate

# Training loop
for epoch in range(1000):
    # forward pass 
    y_pred = X.dot(w)+b
    loss = np.mean((Y - y_pred)**2)
    
    # backward pass (compute gradients)
    dw = -2 * np.mean((Y - y_pred) * X)
    db = -2 * np.mean(Y - y_pred)
    
    #update weights
    w -= lr * dw
    b -= lr * db
    
    if epoch % 100 ==0:
        print(f"Epoch {epoch}, Loss: {loss:.4f}")
        
    # final prediction
    print("Trained Weight:", w)
    print("Trained Bias:", b)