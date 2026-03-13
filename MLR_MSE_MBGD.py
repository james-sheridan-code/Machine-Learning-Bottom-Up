"""
Manually Coding a Multiple Linear Regression.

Model Architecture: Y = (W @ X + b) [no activation function]
Cost Function: MSE (Mean Squared Error)
Optimisation: MBGD (Mini-Batch Gradient Descent)

"""

import numpy as np

def main():
    X_train = np.array([[2104, 5, 1, 45], [1416, 3, 2, 40], [852, 2, 1, 35]])
    y_train = np.array([460, 232, 178])

    X_train = standardise(X_train)

    mini_batch_gradient_descent(W=np.array([0,0,0,0]), b=0, learn_rate=0.01, 
                                iterations=20000, y_train=y_train, X_train=X_train,batch_size = 1)


def standardise(variable):
    mean = np.mean(variable, axis=0)
    std = np.std(variable, axis=0)
    return (variable - mean) / std


def mini_batch_gradient_descent(W, b, learn_rate, iterations, y_train, X_train, batch_size):
    m = len(y_train)
    W = W.reshape(-1,1)
    y_train = y_train.reshape(-1,1)

    for i in range(iterations):
        # shuffle data
        permutation = np.random.permutation(m)
        X_shuffled = X_train[permutation]
        y_shuffled = y_train[permutation]

        for j in range(0, m, batch_size):
            X_batch = X_shuffled[j:j+batch_size]
            y_batch = y_shuffled[j:j+batch_size]

            y_pred = ((X_batch @ W) + b)

            # MSE, and partial derivatives for W and b
            cost = 0.5 * np.mean((y_pred - y_batch)**2)
            dw = (1/batch_size) * (X_batch.T @ (y_pred - y_batch))
            db = (1/batch_size) * np.sum(y_pred - y_batch)

            # update gradients
            temp_W = W - learn_rate * dw
            temp_b = b - learn_rate * db
            W = temp_W
            b = temp_b

            if i%100 == 0:
                print(f"Iteration: {i} Batch {j}\tCost: {cost:.4f}\tW: {W[0,0]:.2f}, {W[1,0]:.2f}, {W[2,0]:.2f}, {W[3,0]:.2f}\tb: {b:.2f}")

    print(f"Final Results:\nw: {W[0,0]:.2f}, {W[1,0]:.2f}, {W[2,0]:.2f}, {W[3,0]:.2f}\nb: {b:.2f}")


if __name__ == '__main__':
    main()