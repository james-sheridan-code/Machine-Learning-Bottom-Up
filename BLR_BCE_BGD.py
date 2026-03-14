"""
Manual implementation of a Binary Linear Regression using NumPy.

Model:          Y = sigmoid(W @ X + b)
Cost Function:  Binary Cross-Entropy (BCE)
Optimisation:   Batch Gradient Descent (BGD)
"""

import numpy as np

def main():
    X_train = np.array([[0.5, 1.5], [1,1], [1.5, 0.5], [3, 0.5], [2, 2], [1, 2.5]]) 
    y_train = np.array([0, 0, 0, 1, 1, 1]) 

    X_train = standardise(X_train)

    W, b = gradient_descent(W=np.zeros(2), b=0, learn_rate=0.01, iterations=10000, 
                            y_train=y_train, X_train=X_train)
    print(f"Final Results:\nw: {W.flatten()}\nb: {b:.2f}")


def standardise(variable):
    """
    Standardise data to have: mean = 0, and standard deviation = 1.
    Used to make gradient descent converge faster and more reliably.
    
    Returns the X matrix with each column standardised. 
    """
    mean = np.mean(variable, axis=0)
    std = np.std(variable, axis=0)
    return (variable - mean) / std


def gradient_descent(W, b, learn_rate, iterations, y_train, X_train):
    """
    Train a binary logistic regression model using Batch Gradient Descent.

    At each iteration, the gradient of the Binary Cross-Entropy (BCE)
    cost function is computed using all training samples, and the
    weights and bias are updated accordingly.

    Returns the final weight matrix (W) and the final bias (b).
    """
    m = X_train.shape[0]
    W = W.reshape(-1,1)
    y_train = y_train.reshape(-1,1)

    for i in range(iterations):
        # get y predictions
        # Formula: y_pred = 1 / [1 + e^-(X_train @ W + b)]
        # Shapes: (6,1) = 1 / [1 + e^-{(6,2) @ (2,1) + scalar}]
        y_pred = 1 / (1 + np.exp(-((X_train @ W) + b)))

        # Binary Cross-Entropy cost function
        cost = -np.mean(y_train * np.log(y_pred) + (1 - y_train) * np.log(1 - y_pred))

        # get partial derivative of w
        # Formula: dw = (1/m) * X.T @ (y_pred - y_train)
        # Shapes: (2,1) = scalar * (6,2).T @ ((6,1) - (6,1))
        dw = (1/m) * (X_train.T @ (y_pred - y_train))
        
        # get partial derivative of b
        # Formula: db = (1/m) * Sum(y_pred - y_train)
        # Shapes: scalar = scalar * scalar
        db = (1/m) * np.sum(y_pred - y_train)

        W -= learn_rate * dw
        b -= learn_rate * db

        if i%100 == 0:
            print(f"Iteration: {i}\tCost: {cost:.4f}\tW: {W.flatten()}\tb: {b:.2f}")
    return W, b


if __name__ == '__main__':
    main()
