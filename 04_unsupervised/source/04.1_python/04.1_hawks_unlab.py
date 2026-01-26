# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt

# Specify the file name
file_path = "Hawks_good.csv"  # Replace with the correct file path if necessary
output_file = "hawks_plot_unlabel.png"  # File name for the saved figure

try:
    # Load the CSV file into a pandas DataFrame
    hawks_data = pd.read_csv(file_path)

    # Plotting
    plt.figure(figsize=(4, 4))  # Set the figure size to 4 inches wide
    plt.scatter(hawks_data['wing'], hawks_data['weight'], alpha=0.7,color="black")

    # Customize the plot

    plt.xlabel("Wing (mm)")
    plt.ylabel("Weight (g)")
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
