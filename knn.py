import numpy as np
import pandas as pd
from collections import Counter
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load the Iris dataset
iris = load_iris()
X = iris.data  # Features
y = iris.target  # Labels

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.45, random_state=42)

# k-NN Algorithm Implementation
def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2)**2))

def knn(X_train, y_train, X_test, k):
    predictions = []
    for test_point in X_test:
        # Calculate distances from the test point to all training points
        distances = [euclidean_distance(test_point, train_point) for train_point in X_train]
        # Get indices of the k smallest distances
        k_indices = np.argsort(distances)[:k]
        # Get the labels of the k nearest neighbors
        k_nearest_labels = [y_train[i] for i in k_indices]
        # Get the most common class label among the k nearest neighbors
        most_common = Counter(k_nearest_labels).most_common(1)
        predictions.append(most_common[0][0])
    return predictions

# Setting k value
k = 8

# Make predictions using k-NN
predictions = knn(X_train, y_train, X_test, k)

# Compare predictions with actual labels and count correct/wrong predictions
correct_predictions = 0
wrong_predictions = 0
for true, predicted in zip(y_test, predictions):
    if true == predicted:
        correct_predictions += 1
    else:
        wrong_predictions += 1

# Calculate accuracy
accuracy = (correct_predictions / len(y_test)) * 100

# Print results
print(f"Correct predictions: {correct_predictions}")
print(f"Wrong predictions: {wrong_predictions}")
print(f"Accuracy: {accuracy:.2f}%")

# Print sample predictions and their corresponding true labels
print("\nSample predictions:")
for true, predicted, features in zip(y_test, predictions, X_test):
    print(f"Features: {features}, True Label: {true}, Predicted Label: {predicted}")
