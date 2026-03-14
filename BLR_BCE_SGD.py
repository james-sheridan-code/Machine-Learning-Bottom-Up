"""
Manual implementation of Binary Logistic Regression using NumPy.

Model:          y = sigmoid(XW + b)
Cost Function:  Binary Cross-Entropy (BCE)
Optimisation:   Stochastic Gradient Descent (SGD)
"""

import numpy as np

def main():
    X_train = np.array([[0.5, 1.5], [1,1], [1.5, 0.5], [3, 0.5], [2, 2], [1, 2.5]]) 
    y_train = np.array([0, 0, 0, 1, 1, 1])

    X_train = standardise(X_train)

    W, b = stochastic_gradient_descent(W=np.zeros(2), b=0.0, learn_rate=0.01, epochs=10000, 
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
    Train a Binary Logistic Regression model using Stochastic Gradient Descent.

    The training data is shuffled at the start of each epoch and the gradients
    are calculated and updated after each row of data in the epoch. The loss 
    function and partial derivatives are calculated from the Binary Cross-Entropy
    (BCE) loss function.

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
            
            # y_pred = 1 / [1 + e^-(X_train @ W + b)]
            # (1,1) = 1 / [1 + e^-{(1,2) @ (2,1) + scalar}]
            y_pred = 1 / (1 + np.exp(-((X_line @ W) + b)))

            # (BCE loss function and partial derivatives for W and b)
            # (1,1) = -((1,1) * log(1,1) + (scalar - (1,1)) * log(scalar - (1,1)))
            epoch_loss += float(-(y_line * np.log(y_pred) + (1 - y_line) * np.log(1 - y_pred)))
            # (1,2) = (1,2) * ((1,1) - (1,1))
            dw = (X_line * (y_pred - y_line))
            # (1,1) = (1,1) - (1,1)
            db = (y_pred - y_line)

            # (update gradients)
            # (2,1) -= scalar * (1,2).T
            W -= learn_rate * dw.T
            # scalar -= scalar * (1,1)
            b -= float(learn_rate * db)

        if i%100 == 0:
            print(f"Iteration: {i} \tAve Cost: {epoch_loss/m:.4f}\tW: {W.flatten()}\tb: {b:.2f}")
    return W, b
    

if __name__ == '__main__':
    main()