import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Read the clustered data
# file_path = 'public_clustered_data.csv'
file_path = 'private_clustered_data.csv'
print(f"Attempting to read file: {file_path}")
df = pd.read_csv(file_path)
print(f"Successfully read data. Shape: {df.shape}")

# Create a figure with multiple scatter plots
#plt.figure(figsize=(15, 10))
plt.figure(figsize=(20, 15))

# Create scatter plots for different feature combinations
feature_pairs = [
    #('1', '2'),
    #('1', '3'),
    #('1', '4'),
    #('2', '3'),
    #('2', '4'),
    #('3', '4')
    ('1', '2'), ('1', '3'), ('1', '4'), ('1', '5'), ('1', '6'),
    ('2', '3'), ('2', '4'), ('2', '5'), ('2', '6'),
    ('3', '4'), ('3', '5'), ('3', '6'),
    ('4', '5'), ('4', '6'),
    ('5', '6')
]


for i, (x_feat, y_feat) in enumerate(feature_pairs, 1):
    # plt.subplot(2, 3, i)
    plt.subplot(5, 3, i)
    # Plot scatter with different colors for each cluster
    sns.scatterplot(data=df, x=x_feat, y=y_feat, hue='Cluster', 
                   palette='husl', alpha=0.5)
    plt.title(f'dimension {x_feat} vs dimension {y_feat}')
    plt.xlabel(f'dimension {x_feat}')
    plt.ylabel(f'dimension {y_feat}')
    
    # Add legend only to the first plot to avoid repetition
    if i != 1:
        plt.legend().remove()

# Adjust layout and save the plot
plt.tight_layout()
# plt.savefig('public_clustered_scatter_plots.png', dpi=300, bbox_inches='tight')
# print("Clustered scatter plots saved as 'public_clustered_scatter_plots.png'")
plt.savefig('private_clustered_scatter_plots.png', dpi=300, bbox_inches='tight')
print("Clustered scatter plots saved as 'private_clustered_scatter_plots.png'")

# Create a separate plot showing the distribution of clusters
plt.figure(figsize=(10, 6))
cluster_counts = df['Cluster'].value_counts().sort_index()
plt.bar(cluster_counts.index, cluster_counts.values)
plt.title('Distribution of Clusters')
plt.xlabel('Cluster')
plt.ylabel('Number of Points')
plt.xticks(cluster_counts.index)
# plt.savefig('cluster_distribution.png', dpi=300, bbox_inches='tight')
# print("Cluster distribution plot saved as 'public_cluster_distribution.png'")
plt.savefig('private_cluster_distribution.png', dpi=300, bbox_inches='tight')
print("Cluster distribution plot saved as 'private_cluster_distribution.png'")