"""
Manual implementation of Multiple Linear Regression using NumPy.

Model:          y = XW + b
Cost Function:  Mean Squared Error (MSE)
Optimisation:   Batch Gradient Descent (BGD)
"""

import numpy as np

def main():
    X_train = np.array([[2104, 5, 1, 45], [1416, 3, 2, 40], [852, 2, 1, 35]])
    y_train = np.array([460, 232, 178])

    X_train = standardise(X_train)

    W, b = batch_gradient_descent(W=np.zeros(4), b=0, learn_rate=0.01, 
                                  iterations=10000, y_train=y_train, X_train=X_train)
    
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


def batch_gradient_descent(W, b, learn_rate, iterations, y_train, X_train):
    """
    Train a linear regression model using Batch Gradient Descent.

    At each iteration, the gradient of the Mean Squared Error (MSE)
    cost function is computed using all training samples, and the
    weights and bias are updated accordingly.

    Returns the final weight matrix (W) and the final bias (b).
    """
    m = X_train.shape[0]
    W = W.reshape(-1,1)
    y_train = y_train.reshape(-1,1)

    for i in range(iterations):
        # Formula: y_pred = X_train @ W + b
        # Shapes: (3,1) = (3,4) @ (4,1) + scalar
        y_pred = ((X_train @ W) + b)

        # MSE Cost Function
        cost = 0.5 * np.mean((y_pred - y_train)**2)

        # Formula: dw = (1/m) * X.T @ (y_pred - y_train)
        # Shapes: (4,1) = scalar * (3,4).T @ ((3,1) - (3,1))
        dw = (1/m) * (X_train.T @ (y_pred - y_train))
        
        # Formula: db = (1/m) * Sum(y_pred - y_train)
        # Shapes: scalar = scalar * scalar
        db = (1/m) * np.sum(y_pred - y_train)

        W -= learn_rate * dw
        b -= learn_rate * db

        if i%100 == 0:
            print(f"Iteration: {i}\tCost: {cost:.4f}\tW: {W[0,0]:.2f}, {W[1,0]:.2f}, {W[2,0]:.2f}, {W[3,0]:.2f}\tb: {b:.2f}")
    return W, b


if __name__ == '__main__':
    main()
