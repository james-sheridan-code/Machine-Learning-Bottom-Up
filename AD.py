"""
Anomaly Detection 
"""

import numpy as np

def main():
    x = np.random.normal(0,1,(100,3))
    
    normal_point = np.array([0.1, -0.2, 0.05])
    anomaly_point = np.array([5, 0, 0])

    mean = np.mean(x, axis=0)
    var = np.var(x, axis=0)

    epsilon = 0.001

    p = probability_estimation(anomaly_point, mean, var)
    print(f"Density: {p:.10f}")
    if p < epsilon:
        print("Anomaly detected!")
    else:
        print("Point is normal.")


def probability_estimation(x, mean, var):
    """The product of probability densities for a data point"""
    part1 = 1 / (np.sqrt(2 * np.pi * var))
    part2 = np.exp(-((x - mean)**2) / (2 * var))
    probabilities = part1 * part2
    return np.prod(probabilities)


if __name__ == '__main__':
    main()