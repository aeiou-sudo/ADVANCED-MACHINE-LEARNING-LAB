import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.preprocessing import StandardScaler

# Load the dataset (replace 'your_data.csv' with the actual path to the dataset)
data = pd.read_csv('generated_dataset.csv')

# Preprocess data: We need to scale the data for better clustering performance
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

# Apply K-Means clustering
kmeans = KMeans(n_clusters=3, random_state=42)  # Change n_clusters as required
kmeans_labels = kmeans.fit_predict(data_scaled)

# Apply EM (Gaussian Mixture Model) clustering
gmm = GaussianMixture(n_components=3, random_state=42)  # Change n_components as required
gmm_labels = gmm.fit_predict(data_scaled)

# Compare the clustering results using silhouette score
kmeans_silhouette = silhouette_score(data_scaled, kmeans_labels)
gmm_silhouette = silhouette_score(data_scaled, gmm_labels)

# Compare the clustering results using Adjusted Rand Index (ARI)
# You will need true labels (ground truth) to compute ARI, which may not be available.
# If you have ground truth labels, replace 'true_labels' with the actual column containing labels
# true_labels = data['true_labels']  # Uncomment and provide your ground truth labels if available
# kmeans_ari = adjusted_rand_score(true_labels, kmeans_labels)
# gmm_ari = adjusted_rand_score(true_labels, gmm_labels)

# Visualize the clusters
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.scatter(data_scaled[:, 0], data_scaled[:, 1], c=kmeans_labels, cmap='viridis')
plt.title('K-Means Clustering')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')

plt.subplot(1, 2, 2)
plt.scatter(data_scaled[:, 0], data_scaled[:, 1], c=gmm_labels, cmap='viridis')
plt.title('EM (Gaussian Mixture) Clustering')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')

plt.show()

# Print the comparison results
print(f'K-Means Silhouette Score: {kmeans_silhouette:.4f}')
print(f'EM (Gaussian Mixture) Silhouette Score: {gmm_silhouette:.4f}')

# Uncomment if you have ground truth labels to compare ARI
# print(f'K-Means Adjusted Rand Index (ARI): {kmeans_ari:.4f}')
# print(f'EM (Gaussian Mixture) Adjusted Rand Index (ARI): {gmm_ari:.4f}')
