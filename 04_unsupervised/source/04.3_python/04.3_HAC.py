import pandas as pd
import scipy.cluster.hierarchy as sch
import matplotlib.pyplot as plt

# Load the dataset (replace 'dslanguages.csv' with your file path)
data = pd.read_csv("dslanguages.csv", index_col=0)

# Perform hierarchical agglomerative clustering
# Calculate the distance matrix (if needed, ensure it's a square matrix)
#linked = sch.linkage(data, method='ward')  # 'ward' minimizes variance; other options: 'single', 'complete', 'average'
linked = sch.linkage(data, method='single')

num_clusters = 7  # Change this number as needed
max_d = linked[-num_clusters, 2]  # Get the threshold distance


# Plot the dendrogram
plt.figure(figsize=(5, 8))
sch.dendrogram(
    linked,
    labels=data.index.tolist(),  # Use language names as labels
    orientation='left',
    distance_sort='descending',
    show_leaf_counts=True,
    color_threshold=max_d
)

# Add title and labels
plt.title('')
plt.xlabel('')
plt.ylabel('')

#plt.savefig("dendrogram_big.png", dpi=300, bbox_inches='tight')
plt.savefig("dendrogram_big_single.png", dpi=300, bbox_inches='tight')

# Show the plot
plt.show()
