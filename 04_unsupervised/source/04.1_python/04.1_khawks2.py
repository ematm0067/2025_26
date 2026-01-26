#the "2" in the file name is for historical reasons
#this does k=3 clustering with L^2

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances_argmin

# Load the Hawk dataset into a DataFrame
df = pd.read_csv('Hawks_good.csv')

# Display the first few rows to understand the structure
print(df.head())

# Select the 'wing' and 'weight' features for clustering
X = df[['wing', 'weight']].dropna().values  # Ensure no missing values

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Function to plot the clusters and centroids
# Function to plot the clusters and centroids
def plot_clusters(ax, X, centroids, labels, iteration, position):
    colors = ['r', 'g', 'b']
    for i in range(3):
        points = X[labels == i]
        ax.scatter(points[:, 0], points[:, 1], s=50, c=colors[i], label=f'Cluster {i+1}')
    ax.scatter(centroids[:, 0], centroids[:, 1], s=200, c='yellow', marker='*', edgecolor='k', label='Centroids')
    
    # Set labels only for leftmost and bottom plots
    if position % 2 == 0:  # Leftmost plots (0 and 2 in 0-based index)
        ax.set_ylabel('Weight (standardized)')
    if position >= 2:  # Bottom plots (2 and 3 in 0-based index)
        ax.set_xlabel('Wing (standardized)')
    
    ax.set_title(f'Iteration {iteration}')
    #ax.legend()

# Randomly initialize centroids by selecting 3 random data points
np.random.seed(42)
#initial_indices = np.random.choice(X_scaled.shape[0], 3, replace=False)
#centroids = X_scaled[initial_indices]
centroids = np.random.uniform(low=-0.25, high=0.25, size=(3, 2))



# Create a 2x2 grid for plotting
fig, axes = plt.subplots(2, 2, figsize=(5, 5))
axes = axes.flatten()

# Perform K-Means clustering for 4 iterations
for iteration in range(4):
    # Step 1: Assign labels based on closest centroid
    labels = pairwise_distances_argmin(X_scaled, centroids)
    
    # Step 2: Plot the clusters and centroids
    plot_clusters(axes[iteration], X_scaled, centroids, labels, iteration + 1, iteration)

    
    # Step 3: Update centroids by computing the mean of assigned points
    new_centroids = np.array([X_scaled[labels == i].mean(axis=0) for i in range(3)])
    
    # Check for convergence (if centroids do not change)
    if np.all(centroids == new_centroids):
        print(f'Convergence reached at iteration {iteration + 1}.')
        break
    
    centroids = new_centroids



output_filename = '04.1_khawks.png'

# Save the figure
plt.savefig(output_filename, dpi=300, bbox_inches='tight')

# Adjust layout and show the composite figure
plt.tight_layout()
plt.show()
