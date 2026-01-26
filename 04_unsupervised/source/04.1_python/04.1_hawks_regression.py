# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
import numpy as np

# Specify the file name
file_path = "Hawks_good.csv"  # Replace with the correct file path if necessary
output_file = "hawks_plot.png"  # File name for the saved figure

try:
    # Load the CSV file into a pandas DataFrame
    hawks_data = pd.read_csv(file_path)

    # Prepare the data for classification
    X = hawks_data[['wing', 'weight']].values  # Features: wing and weight
    y = hawks_data['species'].astype('category').cat.codes  # Encode species as integers

    # Fit a logistic regression model
    model = LogisticRegression(multi_class='ovr')  # One-vs-rest for multiple classes
    model.fit(X, y)

    # Plotting
    plt.figure(figsize=(4, 4))  # Set the figure size to 4 inches wide

    # Plot the data points
    for species, group in hawks_data.groupby('species'):
        plt.scatter(group['wing'], group['weight'], label=species, alpha=0.7)

    # Plot decision boundaries
    x_min, x_max = hawks_data['wing'].min() - 10, hawks_data['wing'].max() + 10
    y_min, y_max = hawks_data['weight'].min() - 100, hawks_data['weight'].max() + 100
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    plt.contour(xx, yy, Z, levels=np.arange(len(np.unique(y)) + 1) - 0.5, colors='k', linestyles='--')

    # Customize the plot
    plt.xlabel("Wing (mm)")
    plt.ylabel("Weight (g)")
    plt.legend(title="Species", fontsize=8)
    plt.grid(True)

    # Save the figure
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"Figure saved as '{output_file}'")

    # Show the plot
    plt.show()

except FileNotFoundError:
    print(f"Error: The file '{file_path}' was not found. Please check the file name and path.")
except pd.errors.EmptyDataError:
    print(f"Error: The file '{file_path}' is empty or invalid.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
