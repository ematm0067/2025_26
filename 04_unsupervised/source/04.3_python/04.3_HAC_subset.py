import pandas as pd
import scipy.cluster.hierarchy as sch
import matplotlib.pyplot as plt

# Load the dataset (replace 'dslanguages.csv' with your file path)
data = pd.read_csv("dslanguages.csv", index_col=0)

# Specify a subset of languages for the smaller tree
subset_languages = ["IrishA","WelshN","BretonSE","RumanianList","French","Spanish","Brazilian","GermanST","Afrikaans","Danish","IcelandicST","EnglishST","LithuanianST","Czech","Slovak","Ukrainian","Polish","Bulgarian","PersianList","AlbanianG"]

# Filter the dataset to include only the specified languages
subset_data = data.loc[subset_languages, subset_languages]

# Perform hierarchical agglomerative clustering on the subset
linked = sch.linkage(subset_data, method='average')  # 'ward' minimizes variance

# Plot the dendrogram for the subset
plt.figure(figsize=(8, 6))
sch.dendrogram(
    linked,
    labels=subset_data.index.tolist(),  # Use language names as labels
    orientation='left',
    distance_sort='descending',
    show_leaf_counts=True
)

# Add title and labels
plt.title('Hierarchical Agglomerative Clustering Dendrogram (Subset)')
plt.xlabel('Languages')
plt.ylabel('Distance')
plt.tight_layout()

# Show the plot
plt.show()
