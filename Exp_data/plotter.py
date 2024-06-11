import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import ast
import numpy as np
# Replace 'file_path.csv' with the path to your CSV file
# file_path = '47.csv'


csv_folder = 'data/06-09_12-53'
plot_folder = './plots'

# Create the plot folder if it doesn't exist
os.makedirs(plot_folder, exist_ok=True)


# Function to read CSV files and create plots
def plot_specific_csv_files(csv_folder, plot_folder, file_names, x_key, y_key, custom_labels):
    csv_folder_path = os.path.join(os.path.dirname(__file__), csv_folder)
    plot_folder_path = os.path.join(os.path.dirname(__file__), plot_folder)

    plt.figure(figsize=(10, 6))

    for i, filename in enumerate(file_names):
        # Read the CSV file
        filepath = os.path.join(csv_folder_path, filename)
        data = pd.read_csv(filepath)
        
        # Check if the required columns are present
        if x_key in data.columns and y_key in data.columns:
            # Plot the data
            sns.lineplot(x=data[x_key], y=data[y_key], label=custom_labels[i])
        else:
            print(f'Columns {x_key} and/or {y_key} not found in {filename}')

    plt.title(f'{"CNN, CIFAR 10, 5 agents, each agent 2 labels"}')
    plt.xlabel(x_key)  # Label for the x-axis
    plt.ylabel(y_key)  # Label for the y-axis

    # Save the plot
    plot_filename = f'plot_{"_".join([os.path.splitext(f)[0] for f in file_names])}.png'
    plot_filepath = os.path.join(plot_folder_path, plot_filename)
    plt.legend()
    #plt.savefig(plot_filepath)
    plt.show()
    plt.close()
    print(f'Saved plot for {", ".join(file_names)} to {plot_filepath}')

# List of specific CSV files to plot
specific_files = ['2_DPFL.csv']
# Columns to plot
x_column = 'iteration'
y_column = 'test_acc'
custom_labels = ['RL aggr']   # , 'Decentralized Average (DSpodFL)', 'Local SGD'

# Call the function to plot specific CSV files
plot_specific_csv_files(csv_folder, plot_folder, specific_files, x_column, y_column, custom_labels)


# Read the CSV file into a Pandas DataFrame
""" df = pd.read_csv(data_path)


plt.figure(figsize=(10, 6))  # Setting figure size

# plot data from a single metric table, e.g., itr and acc; or plot data points from multiple metric tables with different parameters, e.g., convergence speed when num of agents varies
sns.lineplot(data=df, x='iteration', y='rewards', hue='hue_column', ci='sd')  # ci='sd' for confidence intervals

plt.title('Line Plot with Confidence Intervals')  # Adding title
plt.xlabel('Iteration')  # Adding x-axis label
plt.ylabel('Reward')  # Adding y-axis label
plt.legend(title='Hue')  # Adding legend
plt.show()  """


csv_folder_path = os.path.join(os.path.dirname(__file__), csv_folder)
filepath = os.path.join(csv_folder_path, '2.csv')
df = pd.read_csv(filepath)



# Extract the last row from the 'actions' column
data_str = df['actions'].iloc[-1]

data_str = data_str.replace('[', '').replace(']', '').replace(',', '')

# Split the string into individual elements and convert to float
data_list = [float(x) for x in data_str.split()]

# Convert the list into a numpy array and reshape into a 10x10 matrix
matrix_data = np.array(data_list).reshape(10, 10)
normalized_matrix_data = matrix_data / matrix_data.sum(axis=1, keepdims=True)

print(matrix_data)



plt.figure(figsize=(10, 8))
sns.heatmap(normalized_matrix_data, cmap='viridis', annot=True, fmt=".2f", linewidths=.5)
plt.title('Heatmap of RL aggregation weight')
plt.xlabel('Columns')
plt.ylabel('Rows')
plt.show()