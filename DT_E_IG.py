"""
Manual implementation of a Decision Tree using NumPy.

Impurity Measure:    Entropy function
Splitting Criterion: Information Gain
"""

import numpy as np
from pprint import pprint

def main():
    X_train = np.array([
        [1, 1, 1, 0, 0, 0], [0, 0, 1, 1, 1, 0], [0, 1, 0, 1, 0, 1], [1, 0, 1, 0, 1, 1],
        [1, 1, 1, 1, 1, 0], [1, 1, 0, 0, 0, 0], [0, 0, 0, 1, 1, 1], [1, 1, 0, 1, 0, 0],
        [0, 1, 0, 0, 1, 0], [1, 0, 1, 1, 0, 1], [0, 1, 1, 0, 0, 0], [1, 0, 0, 1, 1, 0],
        [1, 1, 0, 1, 1, 1], [0, 0, 1, 0, 0, 0], [0, 1, 1, 1, 1, 1], [1, 1, 1, 0, 1, 0],
        [0, 0, 0, 0, 0, 1], [1, 0, 1, 1, 1, 0], [1, 1, 0, 0, 1, 1], [0, 1, 0, 1, 0, 0]
    ])
    y_train = np.array([1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 0])

    tree = build_tree(X_train, y_train, current_depth=0, max_depth=5)
    print("\nTree Structure:\n")
    pprint(tree, indent=4)
    
    features = [0,0,0,1,0,1]
    prediction = predict(tree, features)
    print(f"\nInput features for prediction: {features}\nPrediction: {prediction}\n")


def entropy(p):
    """
    Entropy measures the impurity of a set of values.
    Pure = 0
    Impure = 1

    Returns the purity value (range: 0 - 1).
    """
    if p == 0 or p == 1:
        return 0
    else:
        return -p * np.log2(p) - (1- p)*np.log2(1 - p)
    

def weighted_average_entropy(X, y, left_branch, right_branch):
    """
    Returns the weighted average of the branch pair.
    """
    if len(left_branch) > 0:
        left_fraction = sum(y[left_branch]) / len(left_branch)
        left_entropy = entropy(left_fraction)
    else:
        left_entropy = 0

    if len(right_branch) > 0:
        right_fraction = sum(y[right_branch]) / len(right_branch)
        right_entropy = entropy(right_fraction)
    else:
        right_entropy = 0

    left_w = len(left_branch)/len(X)
    right_w = len(right_branch)/len(X)

    return (left_w * left_entropy) + (right_w * right_entropy)


def info_gain(X, y, left_branch, right_branch):
    """
    Returns the information gain of a branch (parent node entropy - weighted entropy).
    """
    parent_node = sum(y)/len(y)
    parent_node_entropy = entropy(parent_node)
    weighted_entropy = weighted_average_entropy(X, y, left_branch, right_branch)
    return parent_node_entropy - weighted_entropy


def split_feature(X, feature):
    """
    Split feature depending on the feature given.

    Returns a list of value's indices in the left and right branch.
    """
    left_branch = []
    right_branch = []
    for i,x in enumerate(X):
        if x[feature] == 1:
            left_branch.append(i)
        else:
            right_branch.append(i)
    return left_branch, right_branch


def build_tree(X, y, current_depth=0, max_depth=5):
    """
    Recursive function continuously building the tree.

    Stopping conditions: pure node, max tree depth met, information gain < 0.1

    Returns a nested dictionary of all the branches.
    """
    prediction = 1 if sum(y) / len(y) >= 0.5 else 0

    if len(np.unique(y)) == 1: # stop if a pure node
        return prediction
    elif current_depth >= max_depth: # stop if tree depth limit is hit
        return prediction

    info_gain_list = []
    for i in range(len(X[0])): 
        left_indices, right_indices = split_feature(X, i)
        i_gain = info_gain(X, y, left_indices, right_indices) 
        info_gain_list.append(i_gain) 

    max_index = np.argmax(info_gain_list) 

    if info_gain_list[max_index] < 0.1: # stop if info gain < limit
        return prediction

    left_indices, right_indices = split_feature(X, max_index)

    left_branch = build_tree(X[left_indices], y[left_indices], current_depth + 1, max_depth)
    right_branch = build_tree(X[right_indices], y[right_indices], current_depth + 1, max_depth)

    return {'feature_idx': max_index, 'left': left_branch, 'right': right_branch}


def predict(tree, features):
    """
    Returns the leaf node value of the tree given the tree and feature values.
    """
    # if at a leaf, read the leaf
    if not isinstance(tree, dict):
        return tree
    # if not at a leaf, follow the tree
    feature_idx = tree['feature_idx']
    if features[feature_idx] == 1:
        return predict(tree['left'], features)
    else:
        return predict(tree['right'], features)


if __name__ == "__main__":
    main()