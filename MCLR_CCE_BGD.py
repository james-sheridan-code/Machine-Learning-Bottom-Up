"""
Manual implementation of Multiclass Logistic Regression using NumPy.

Model:          y = softmax(XW + b)
Cost Function:  Categorical Cross-Entropy (CCE)
Optimisation:   Batch Gradient Descent (BGD)
"""

import numpy as np

def main():
    X_train = np.array([[0.5, 1.5], [1.0, 1.0], [1.5, 0.5], [3.0, 0.5], 
                        [2.5, 1.0], [3.5, 1.0], [1.0, 3.0], [1.5, 2.5], [0.5, 2.5]])
    y_one_hot = np.array([[1, 0, 0], [1, 0, 0], [1, 0, 0], [0, 1, 0],
                        [0, 1, 0], [0, 1, 0], [0, 0, 1], [0, 0, 1], [0, 0, 1]])

    X_train = standardise(X_train)

    W, b = batch_gradient_descent(W=np.zeros((X_train.shape[-1],3)), b=np.zeros((1,3)), learn_rate=0.01, iterations=10000, 
                            y_train=y_one_hot, X_train=X_train)
    
    print(f"Final Results:\nw: {W.flatten()}\nb: {b.flatten()}")


def standardise(variable):
    """
    Standardise data to have: mean = 0, and standard deviation = 1.
    Used to make gradient descent converge faster and more reliably.
    
    Returns the X matrix with each column standardised. 
    """
    mean = np.mean(variable, axis=0)
    std = np.std(variable, axis=0)
    return (variable - mean) / std


def softmax(Z):
    exp_Z = np.exp(Z - np.max(Z, axis=1, keepdims=True))
    return exp_Z / np.sum(exp_Z, axis=1, keepdims=True)


# everything needs to change in here!!
def batch_gradient_descent(W, b, learn_rate, iterations, y_train, X_train):
    """
    Train a binary logistic regression model using Batch Gradient Descent.

    At each iteration, the gradient of the Categorical Cross-Entropy (CCE)
    cost function is computed using all training samples, and the
    weights and bias are updated accordingly.

    Returns the final weight matrix (W) and the final bias (b).
    """
    m = X_train.shape[0]

    for i in range(iterations):
        # get y predictions (with softmax)
        Z = X_train @ W + b
        y_pred = softmax(Z)

        # Categorical Cross-Entropy cost function
        cost = -np.sum(y_train * np.log(y_pred + 1e-9)) / m

        # get partial derivative of w
        dw = (1/m) * (X_train.T @ (y_pred - y_train))
        
        # get partial derivative of b
        db = (1/m) * np.sum(y_pred - y_train, axis=0, keepdims=True)

        W -= learn_rate * dw
        b -= learn_rate * db

        if i%100 == 0:
            print(f"Iteration: {i}\tCost: {cost:.4f}\tW: {W.flatten()}\tb: {b.flatten()}")
    return W, b


if __name__ == '__main__':
    main()