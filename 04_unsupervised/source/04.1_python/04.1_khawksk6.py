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
def plot_clusters(ax, X, centroids, labels, iteration, position):
    colors = ['r', 'g', 'b', 'c', 'm', 'y']  # Add more colors for additional clusters
    for i in range(6):
        points = X[labels == i]
        ax.scatter(points[:, 0], points[:, 1], s=50, c=colors[i])
        ax.scatter(centroids[:, 0], centroids[:, 1], s=200, c='yellow', marker='*', edgecolor='k')
    #ax.legend()
    # Set labels only for leftmost and bottom plots
    if position % 3 == 0:  # Leftmost plots (0 and 2 in 0-based index)
        ax.set_ylabel('Weight (stdized)')
    if position >= 6:  # Bottom plots (2 and 3 in 0-based index)
        ax.set_xlabel('Wing (stdized)')
    
    ax.set_title(f'Iteration {iteration}')
    #ax.legend()

# Initialize centroids within the specified range
np.random.seed(42)
initial_indices = np.random.choice(X_scaled.shape[0], 6, replace=False)
centroids = X_scaled[initial_indices]


# Create a 2x2 grid for plotting
fig, axes = plt.subplots(3, 3, figsize=(5, 5))
axes = axes.flatten()

# Perform K-Means clustering for 4 iterations
for iteration in range(9):
    # Step 1: Assign labels based on closest centroid
    labels = pairwise_distances_argmin(X_scaled, centroids)
    
    # Step 2: Plot the clusters and centroids
    plot_clusters(axes[iteration], X_scaled, centroids, labels, iteration + 1, iteration)
    
    # Step 3: Update centroids by computing the mean of assigned points
    new_centroids = np.array([X_scaled[labels == i].mean(axis=0) for i in range(6)])
    
    # Check for convergence (if centroids do not change)
    #if np.all(centroids == new_centroids):
    #    print(f'Convergence reached at iteration {iteration + 1}.')
    #    break
    
    centroids = new_centroids

# Adjust layout
plt.tight_layout()

# Save the figure
plt.savefig('04.1_khawks_k6.png', dpi=300, bbox_inches='tight')

# Display the figure
plt.show()
