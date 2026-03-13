"""
Manually Coding a Simple Linear Regression.

Model Architecture: y = (w * x + b)
Cost Function: MSE (Mean Squared Error)
Optimisation: BGD (Batch Gradient Descent)

"""

import numpy as np

def main():
    # data
    x_train = np.array([1.0, 2.0])
    y_train = np.array([300.0, 500.0])

    # initialise
    w = 0
    b = 0
    learn = 0.01
    iterations = 50000

    # run gradient decent
    gradient_descent(w, b, learn, iterations, x_train, y_train)


def gradient_descent(w, b, learn, iterations, x_train, y_train):
    m = len(y_train)
    
    for i in range(iterations):
        y_pred = w * x_train + b

        # MSE Cost Function
        cost = 0.5 * np.mean((y_pred - y_train)**2)

        # derivatives
        dw = (1/m) * np.sum((y_pred - y_train) * x_train)
        db = (1/m) * np.sum(y_pred - y_train)

        w -= learn * dw
        b -= learn * db

        if i%100 == 0:
            print(f"Iteration: {i}\tcost: {cost}\tw: {w:.2f}\tb: {b:.2f}\tdw:{dw:.2f}\tdb:{db:.2f}")
    
    print(f"Final w: {w:.2f}\nFinal b: {b:.2f}\nFinal Equation: y = {b:.2f} + {w:.2f}x")


if __name__ == '__main__':
    main()