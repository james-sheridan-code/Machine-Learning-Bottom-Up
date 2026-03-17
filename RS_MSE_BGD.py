"""
Manually coding a recommender system.

"""

import numpy as np

def main():
    # data
    X_user = np.array([
        [1.0, 0.1, 0.2, 0.9], # User A
        [0.9, 0.2, 0.1, 0.8], # User B
        [0.8, 0.3, 0.2, 0.9], # User C
        [0.1, 0.9, 0.8, 0.2], # User D
        [0.2, 0.8, 0.9, 0.1], # User E
        [0.1, 0.7, 0.8, 0.2]  # User F
    ])
    X_product = np.array([
        [0.9, 0.2, 0.1, 1.0], # Product 1 (Action?)
        [1.0, 0.1, 0.2, 0.9], # Product 2
        [0.8, 0.2, 0.1, 0.8], # Product 3
        [0.2, 0.9, 0.8, 0.1], # Product 4 (Romance?)
        [0.1, 0.8, 0.9, 0.2], # Product 5
        [0.2, 0.7, 0.9, 0.1]  # Product 6
    ])

    # Ratings (1 to 5 scale)
    Y = np.array([5.0, 4.8, 4.5, 1.2, 1.5, 1.0])

    X_user = standardise(X_user)
    X_product = standardise(X_product)

    # initialising 2 layers of 2 NN: 5 neurons, 3 neurons each
    np.random.seed(0)
    params = {
        "W1_u": np.random.randn(5, 4) * 0.1,
        "b1_u": np.zeros((5, 1)),
        "W2_u": np.random.randn(3, 5) * 0.1,
        "b2_u": np.zeros((3, 1)),
        "W1_p": np.random.randn(5, 4) * 0.1,
        "b1_p": np.zeros((5, 1)),
        "W2_p": np.random.randn(3, 5) * 0.1,
        "b2_p": np.zeros((3, 1))
    }

    learning_rate = 0.05
    iterations = 5000
    params = batch_gradient_descent(X_user, X_product, Y, params, learning_rate, iterations)

    print(f"Final values: \n{params}")
    

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


def forward_prop_NN1(X_train, W1, b1, W2, b2):
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
    A2 = Z2

    # store outputs for backpropagation
    cache_u = {"NN1Z1": Z1, "NN1A1": A1, "NN1Z2": Z2, "NN1A2": A2}
    return A2, cache_u


def forward_prop_NN2(X_train, W1, b1, W2, b2):
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
    A2 = Z2

    # store outputs for backpropagation
    cache_p = {"NN2Z1": Z1, "NN2A1": A1, "NN2Z2": Z2, "NN2A2": A2}
    return A2, cache_p


def backward_prop(X, dA2, cache, W2, prefix):
    """
    Using back propagation to determine the gradients.
    
    X: Input features
    dA2: The gradient passed down from the dot product interaction
    cache: The specific cache for this tower (NN1 or NN2)
    W2: The second layer weights for this tower
    prefix: "NN1" or "NN2" to access the right keys in cache
    """
    m = X.shape[0]
    
    # Access cache
    A1 = cache[f'{prefix}A1']
    Z1 = cache[f'{prefix}Z1']
    Z2 = cache[f'{prefix}Z2']
    
    # L2: Linear activation
    dZ2 = dA2 
    dW2 = (1/m) * (dZ2 @ A1.T)
    db2 = (1/m) * np.sum(dZ2, axis=1, keepdims=True)

    # L1: reLU
    dA1 = W2.T @ dZ2
    dZ1 = dA1 * (Z1 > 0)
    dW1 = (1/m) * (dZ1 @ X)
    db1 = (1/m) * np.sum(dZ1, axis=1, keepdims=True)

    return {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}   


def batch_gradient_descent(X_user, X_product, Y, params, learning_rate, iterations):
    """
    Use batch gradient descent to update the weights and biases of both neural 
    networks using a MSE cost function using both neural networks.
    """
    m = Y.shape[0]

    for i in range(iterations):
        # get the final vectors and their weights and bias
        Vu, cache_u = forward_prop_NN1(X_user, params["W1_u"], params["b1_u"], params["W2_u"], params["b2_u"])
        Vp, cache_p = forward_prop_NN2(X_product, params["W1_p"], params["b1_p"], params["W2_p"], params["b2_p"])

        y_hat = np.sum(Vu * Vp, axis=0)

        # MSE cost function
        error = y_hat - Y
        cost = (1 / (2 * m)) * np.sum(np.square(error))

        # partial derivatives relative to loss function
        error_reshaped = error.reshape(1, -1)
        dA2_u = error_reshaped * Vp 
        dA2_p = error_reshaped * Vu

        # get gradients
        grad_u = backward_prop(X_user, dA2_u, cache_u, params["W2_u"], prefix="NN1")
        grad_p = backward_prop(X_product, dA2_p, cache_p, params["W2_p"], prefix="NN2")

        # update parameters
        params["W1_u"] -= learning_rate * grad_u["dW1"]
        params["b1_u"] -= learning_rate * grad_u["db1"]
        params["W2_u"] -= learning_rate * grad_u["dW2"]
        params["b2_u"] -= learning_rate * grad_u["db2"]

        params["W1_p"] -= learning_rate * grad_p["dW1"]
        params["b1_p"] -= learning_rate * grad_p["db1"]
        params["W2_p"] -= learning_rate * grad_p["dW2"]
        params["b2_p"] -= learning_rate * grad_p["db2"]

        if i%100 == 0:
            print(f"Iteration {i}'s cost: {cost}.")

    return params


if __name__ == '__main__':
    main()