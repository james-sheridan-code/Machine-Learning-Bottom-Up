"""
Manually do Binary Logistic Regression using gradient descent.

Steps in Binary Logistic Regression:
1. Standardise
(loop over):
2. y_prediction matrix
3. get both partial derivatives
4. replace b and w for gradient descent.
"""


import numpy as np

def main():
    X_train = np.array([[0.5, 1.5], [1,1], [1.5, 0.5], [3, 0.5], [2, 2], [1, 2.5]]) 
    y_train = np.array([0, 0, 0, 1, 1, 1]) 

    X_train = standardise(X_train)

    gradient_descent(W=np.array([0.0,0.0]), b=0, learn_rate=0.01, iterations=10000, y_train=y_train, X_train=X_train)


def standardise(variable):
    mean = np.mean(variable, axis=0)
    std = np.std(variable, axis=0)
    return (variable - mean) / std


def gradient_descent(W, b, learn_rate, iterations, y_train, X_train):

    m = len(y_train)
    W = W.reshape(-1,1)
    y_train = y_train.reshape(-1,1)

    for i in range(iterations):
        # get y predictions
        # Formula: y_pred = 1 / [1 + e^-(X_train @ W + b)]
        # Shapes: (6,1) = 1 / [1 + e^-{(6,2) @ (2,1) + scalar}]
        y_pred = 1 / (1 + np.exp(-((X_train @ W) + b)))

        # current cost function (to check if cost goes down over time)
        cost = -np.mean(y_train * np.log(y_pred) + (1 - y_train) * np.log(1 - y_pred))

        # get partial derivative of w
        # Formula: dw = (1/m) * X.T @ (y_pred - y_train)
        # Shapes: (2,1) = scalar * (6,2).T @ ((6,1) - (6,1))
        dw = (1/m) * (X_train.T @ (y_pred - y_train))
        
        # get partial derivative of b
        # Formula: db = (1/m) * Sum(y_pred - y_train)
        # Shapes: scalar = scalar * scalar
        db = (1/m) * np.sum(y_pred - y_train)

        temp_W = W - learn_rate * dw
        temp_b = b - learn_rate * db
        W = temp_W
        b = temp_b

        if i%100 == 0:
            print(f"Iteration: {i}\tCost: {cost:.4f}\tW: {W[0,0]:.2f}, {W[1,0]:.2f}\tb: {b:.2f}")

    print(f"Final Results:\nw: {W[0,0]:.2f}, {W[1,0]:.2f}\nb: {b:.2f}")


if __name__ == '__main__':

    main()
