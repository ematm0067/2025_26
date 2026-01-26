import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Step 1: Generate 5 sets of 2D points, each with 20 values
np.random.seed(42)  # For reproducibility
n_clusters = 5
n_points_per_cluster = 20
sd = 0.5  # Standard deviation of Gaussian

# Generate cluster centers uniformly in the [-5, 5] x [-5, 5] box
centers = np.random.uniform(-5, 5, size=(n_clusters, 2))

# Generate points around each center
points = []
labels = []
for i, center in enumerate(centers):
    cluster_points = np.random.normal(loc=center, scale=sd, size=(n_points_per_cluster, 2))
    points.append(cluster_points)
    labels.extend([i] * n_points_per_cluster)

points = np.vstack(points)  # Combine all points into a single array

# Step 2: Plot the points with different colors for each set
plt.figure(figsize=(5, 5))
for i in range(n_clusters):
    cluster_points = points[labels == i] if isinstance(labels, np.ndarray) else points[np.array(labels) == i]
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f"Cluster {i + 1}")
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Generated Points by Cluster")
plt.legend()
plt.grid()
plt.savefig("04.2_points.png", dpi=300)
plt.close()

# Step 3: Perform k-means for k from 1 to 10 and calculate J
k_values = range(1, 11)
J_values = []

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(points)
    J_values.append(kmeans.inertia_)  # Inertia is the sum of squared distances (J)

# Step 4: Plot J as a function of k
plt.figure(figsize=(5, 5))
plt.plot(k_values, J_values, marker='o')
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Sum of Squared Distances (J)")
plt.title("Elbow Plot: J vs. k")
plt.grid()
plt.savefig("04.2_J.png", dpi=300)
plt.close()
