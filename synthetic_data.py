import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler

# Generate synthetic dataset with 3 clusters and 500 samples
n_samples = 500
n_features = 2
n_clusters = 3

# Generate random data points with 3 clusters
X, y = make_blobs(n_samples=n_samples, centers=n_clusters, cluster_std=0.60, random_state=42)

# Visualize the dataset
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', s=30)
plt.title('Generated Dataset with 3 Clusters')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()

# Standardize the dataset (important for clustering algorithms)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# You can save the dataset as a CSV file if needed
import pandas as pd
dataset = pd.DataFrame(X_scaled, columns=['Feature 1', 'Feature 2'])
dataset.to_csv('generated_dataset.csv', index=False)

print("Synthetic dataset generated and saved as 'generated_dataset.csv'.")
