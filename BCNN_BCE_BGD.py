"""
Manual implementation of Binary Classification Neural Network using NumPy.

Model:          L1: 5 neurons, reLU activation
                L2: 3 neurons, reLU activation
                L3: 1 neuron, sigmoid activation
Cost Function:  Binary Cross-Entropy (BCE)
Optimisation:   Batch Gradient Descent (BGD)
"""

import numpy as np

def main():
    X_train = np.array([[220, 30, 3500, 2],
                        [219, 20, 3400, 3],
                        [240, 21, 3110, 7],
                        [221, 30, 6432, 2],
                        [321, 12, 3200, 9]])
    Y_train = np.array([1,1,0,0,1]).reshape(1,-1)

    X_train = standardise(X_train)

    # initialising 3 layers: 5 neurons, 3 neurons, 1 neuron
    W1 = np.random.randn(5,4) * 0.1
    b1 = np.zeros((5,1))
    W2 = np.random.randn(3,5) * 0.1
    b2 = np.zeros((3,1))
    W3 = np.random.randn(1,3) * 0.1
    b3 = np.zeros((1,1))

    learning_rate = 0.01
    iterations = 20000
    params = batch_gradient_descent(X_train, Y_train, W1, b1, W2, b2, W3, b3, learning_rate, iterations)

    print(f"Final: W1: \n{params['W1']}, \nW2: \n{params['W2']}, \nW3: \n{params['W3']}\nb1: \n{params['b1']}, \nb2: \n{params['b2']}, \nb3: \n{params['b3']}")
    
def standardise(x):
    """
    Standardise data to have: mean = 0, and standard deviation = 1.
    Used to make gradient descent converge faster and more reliably.
    
    Returns the X matrix with each column standardised. 
    """
    mean = np.mean(x, axis=0)
    std = np.std(x, axis=0)
    return (x - mean) / std


def reLU(x):
    """
    reLU activation function, returning the maximum of value 'x' and '0'.
    """
    return np.maximum(0,x)


def sigmoid(x):
    """
    Sigmoid activation function, returning a value between 0 and 1.
    """
    return 1 / (1 + np.exp(-x))


def forward_prop(X_train, W1, b1, W2, b2, W3, b3):
    """
    Forward propagation calculates the final output value of the neural network
    using the given X matrix, W matrices, and b matrices.

    Returns the final output (A3) and cache containing each layer's Z and A value.
    """
    # L1 Shape: (5,m) = (5,4) @ (4,m) + (5,1)
    Z1 = W1 @ X_train.T + b1
    A1 = reLU(Z1)

    # L2 Shape: (3,m) = (3,5) @ (5,m) + (3,1)
    Z2 = W2 @ A1 + b2
    A2 = reLU(Z2)

    # L3 Shape: (1,m) = (1,3) @ (3,m) + (1,1)
    Z3 = W3 @ A2 + b3
    A3 = sigmoid(Z3)

    # store outputs for backpropagation
    cache = {"Z1": Z1, "A1": A1, "Z2": Z2, "A2": A2, "Z3": Z3, "A3": A3}
    return A3, cache


def backward_prop(X, Y, cache, W1, W2, W3):
    """
    Back propagation calculates the 'w' and 'b' partial derivative for each layer
    relative to the Loss function (BinaryCrossEntropy).

    Returns the 'dw' and 'db' partial derivatives for each layer in the neural network.
    """
    m = X.shape[0]
    A1, A2, A3 = cache['A1'], cache['A2'], cache['A3']
    Z1, Z2 = cache['Z1'], cache['Z2']
    
    # L3: (sigmoid)
    dZ3 = A3 - Y # combining dA3 and dZ3 in one step, to not potentially divide by zero
    dW3 = (1/m) * (dZ3 @ A2.T)
    db3 = (1/m) * np.sum(dZ3, axis=1, keepdims=True)

    # L2: (reLU)
    dA2 = W3.T @ dZ3
    dZ2 = dA2 * (Z2 > 0)
    dW2 = (1/m) * dZ2 @ A1.T
    db2 = (1/m) * np.sum(dZ2, axis=1, keepdims=True)

    # L1: (reLU)
    dA1 = W2.T @ dZ2
    dZ1 = dA1 * (Z1 > 0)
    dW1 = (1/m) * dZ1 @ X
    db1 = (1/m) * np.sum(dZ1, axis=1, keepdims=True)

    return {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2, "dW3": dW3, "db3": db3}


def batch_gradient_descent(X_train, Y_train, W1, b1, W2, b2, W3, b3, learning_rate, iterations):
    """
    Train a Neural Network using Batch Gradient Descent.

    At each iteration, the gradient of the Binary Cross-Entropy (BCE)
    cost function is computed using all training samples, and the
    weights and bias are updated accordingly.

    Returns the final W and b matrices.
    """
    for i in range(iterations):
        A3, cache = forward_prop(X_train, W1, b1, W2, b2, W3, b3)
        derivs = backward_prop(X_train, Y_train, cache, W1, W2, W3)
        
        e = 0.00000001
        cost = -np.mean(Y_train * np.log(A3+e) + (1 - Y_train) * np.log(1 - A3+e))

        W1 -= learning_rate * derivs["dW1"]
        W2 -= learning_rate * derivs["dW2"]
        W3 -= learning_rate * derivs["dW3"]
        b1 -= learning_rate * derivs["db1"]
        b2 -= learning_rate * derivs["db2"]
        b3 -= learning_rate * derivs["db3"]

        if i%100 == 0:
            print(f"Iteration {i}'s cost: {cost}.")

    return {"W1": W1,"W2": W2, "W3": W3,"b1": b1,"b2": b2, "b3": b3}


if __name__ == '__main__':
    main()