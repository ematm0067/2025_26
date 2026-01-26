import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances_argmin

metric = 'l1'


# Load the Hawk dataset into a DataFrame
df = pd.read_csv('Hawks_good.csv')

# Display the first few rows to understand the structure
print(df.head())

# Select the 'wing' and 'weight' features for clustering
X = df[['wing', 'weight']].dropna().values  # Ensure no missing values

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

def plot_clusters(ax, X, centroids, labels):
    colors = ['r', 'g', 'b']
    for i in range(3):
        points = X[labels == i]
        ax.scatter(points[:, 0], points[:, 1], s=50, c=colors[i], label=f'Cluster {i+1}')
    ax.scatter(centroids[:, 0], centroids[:, 1], s=200, c='yellow', marker='*', edgecolor='k', label='Centroids')
    ax.set_xlabel('Wing (standardized)')
    ax.set_ylabel('Weight (standardized)')
    #ax.set_title('Final Clustering')
    #ax.legend()

# Randomly initialize centroids by selecting 3 random data points
# for 42 the clustering is correct, for 142 incorrect!

seed=142

np.random.seed(seed)
initial_indices = np.random.choice(X_scaled.shape[0], 3, replace=False)
centroids = X_scaled[initial_indices]

iteration = 0
while True:
    # Step 1: Assign labels based on closest centroid
    labels = pairwise_distances_argmin(X_scaled, centroids, metric=metric)

    # Step 2: Update centroids by computing the mean of assigned points
    new_centroids = np.array([X_scaled[labels == i].mean(axis=0) for i in range(3)])

    # Check for convergence (if centroids do not change)
    if np.all(centroids == new_centroids):
        print(f'Convergence reached at iteration {iteration + 1}.')
        break

    centroids = new_centroids
    iteration += 1

J = sum(np.sum((X_scaled[labels == i] - centroids[i])**2) for i in range(3))
print(J)
    
# Plot the final clustering
fig, ax = plt.subplots(figsize=(2.5, 2.5))
plot_clusters(ax, X_scaled, centroids, labels)

output_filename = '04.1_khawks_' + metric + '.png'
#output_filename = '04.1_khawks_'+str(seed)+'.png'

# Save the figure
plt.savefig(output_filename, dpi=300, bbox_inches='tight')
plt.close()
