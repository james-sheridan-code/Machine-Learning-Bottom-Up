"""
K-means Clustering Algorithm

"""

import numpy as np
import matplotlib.pyplot as plt

def main():
    data = np.array([
        [1.0, 1.0], [1.2, 0.8], [0.8, 1.2],
        [5.0, 5.0], [5.2, 4.8], [4.8, 5.2], 
        [9.0, 1.0], [9.2, 0.8], [8.8, 1.2]
    ])

    K = 3
    centroids = initialise_centroids(K, data)

    iteration = 1
    while True:
        # store old centroids and update the centroids
        print(f"Iteration: {iteration}")
        assignments = assign_to_centroids(data, centroids)
        old_centroids = centroids.copy()
        centroids = adjust_centroids(data, assignments, centroids, K)
        
        # check if there is no more movement
        if (np.array_equal(centroids, old_centroids)):
            print("Convergence reached.")
            break
        iteration += 1

    colors = ['r', 'g', 'b']
    for i in range(K):
        cluster_points = data[assignments == i]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], c=colors[i], marker='x', label=f'Cluster {i+1}')

    # Plot the final centroids
    plt.scatter(centroids[:,0], centroids[:,1], marker='o', c='black', s=150, label='Final Centroids')
    plt.title("Final K-Means Clustering")
    plt.legend()
    plt.show()


def initialise_centroids(K, data):
    """Randomly use a data point to position each centroid"""
    start = np.random.choice(data.shape[0], size=K, replace=False)
    return data[start]


def assign_to_centroids(data, centroids):
    """Assigns each data point to the nearest centroid"""
    assignments = []
    for i in data:
        distances = [np.linalg.norm(i - centroid) for centroid in centroids]
        assignments.append(np.argmin(distances))
    return np.array(assignments)


def adjust_centroids(data, assignments, centroids, K):
    """Updates centroid positions to the mean of their assigned data points."""
    new_centroids = np.zeros_like(centroids)
    
    for i in range(K):
        # Find points assigned to cluster i
        cluster_points = data[assignments == i]
        
        if len(cluster_points) > 0:
            new_centroids[i] = np.mean(cluster_points, axis=0)
        else:
            # If a cluster is empty, keep the centroid where it is
            new_centroids[i] = centroids[i]
            
    return new_centroids


if __name__ == '__main__':
    main()