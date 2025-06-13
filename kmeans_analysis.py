import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import os

# Print current working directory
print(f"Current working directory: {os.getcwd()}")

# Read the data
file_path = 'big data/private_data.csv'
print(f"Attempting to read file: {file_path}")
df = pd.read_csv(file_path)
print(f"Successfully read data. Shape: {df.shape}")

# Extract features (excluding the ID column)
X = df.iloc[:, 1:7].values  # Using all 6 dimensions
print(f"Feature matrix shape: {X.shape}")

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Set number of clusters to 23
n_clusters = 23

# Perform K-means clustering
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_scaled)
print("K-means clustering completed")

# Increment cluster labels by 1 so there is no cluster 0
clusters = clusters + 1

# Add cluster assignments back to the dataframe
df['Cluster'] = clusters

# Calculate cluster statistics
cluster_stats = df.groupby('Cluster').agg({
    '1': ['mean', 'std'],
    '2': ['mean', 'std'],
    '3': ['mean', 'std'],
    '4': ['mean', 'std'],
    '5': ['mean', 'std'],
    '6': ['mean', 'std']
}).round(2)

# Print cluster statistics
print("\nCluster Statistics:")
print(cluster_stats)

# Calculate and print the percentage of data points in each cluster
cluster_sizes = df['Cluster'].value_counts().sort_index()
print("\nCluster Sizes (Percentage of Total Data):")
for cluster, size in cluster_sizes.items():
    percentage = (size / len(df)) * 100
    print(f"Cluster {cluster}: {size} points ({percentage:.2f}%)")

# Calculate silhouette score to evaluate clustering quality
silhouette_avg = silhouette_score(X_scaled, clusters)
print(f"\nSilhouette Score: {silhouette_avg:.3f}")

# Save the clustered data
output_file = 'private_clustered_data.csv'
df.to_csv(output_file, index=False)
print(f"\nClustered data saved to {output_file}") 