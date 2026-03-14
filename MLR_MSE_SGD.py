"""
Manual implementation of Multiple Linear Regression using NumPy.

Model:          y = XW + b
Cost Function:  Mean Squared Error (MSE)
Optimisation:   Stochastic Gradient Descent (SGD)
"""

import numpy as np

def main():
    X_train = np.array([[2104, 5, 1, 45], [1416, 3, 2, 40], [852, 2, 1, 35]])
    y_train = np.array([460, 232, 178])

    X_train = standardise(X_train)

    W, b = stochastic_gradient_descent(W=np.zeros(4), b=0.0, learn_rate=0.01, epochs=10000, 
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


def stochastic_gradient_descent(W, b, learn_rate, epochs, y_train, X_train):
    """
    Train a Multiple Linear Regression model using Stochastic Gradient Descent.

    The training data is shuffled at the start of each epoch and the gradients
    are calculated and updated after each row of data in the epoch. The loss 
    function and partial derivatives are calculated from the Mean Squared
    Error (MSE) loss function.

    Returns the final weight matrix (W) and the final bias (b).
    """
    m = X_train.shape[0]
    W = W.reshape(-1,1)
    y_train = y_train.reshape(-1,1)

    for i in range(epochs):
        # randomise order each epoch
        indices = np.random.permutation(m)
        X_shuffled = X_train[indices]
        y_shuffled = y_train[indices]

        epoch_loss = 0
        
        for j in range(0, m):
            # only look at one row of the batch
            X_line = X_shuffled[[j]]
            y_line = y_shuffled[[j]]
            
            # scalar = (1,4) @ (4,1) + scalar
            y_pred = X_line @ W + b

            # (MSE loss function and partial derivatives for W and b)
            # scalar = 0.5 * (scalar - scalar)**2
            epoch_loss += float(0.5 * (y_pred - y_line)**2)
            # (1,4) = (1,4) * (scalar - scalar)
            dw = (X_line * (y_pred - y_line))
            # (1,1) = (scalar - scalar)
            db = (y_pred - y_line)

            # (update gradients)
            # (4,1) -= scalar * (1,4).T
            W -= learn_rate * dw.T
            # scalar -= scalar * (1,1)
            b -= float(learn_rate * db)

        if i%100 == 0:
            print(f"Iteration: {i} \tAve Cost: {epoch_loss/m:.4f}\tW: {W.flatten()}\tb: {b:.2f}")
    return W, b
    

if __name__ == '__main__':
    main()