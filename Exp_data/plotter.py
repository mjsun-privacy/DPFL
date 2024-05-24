import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Replace 'file_path.csv' with the path to your CSV file
# file_path = '47.csv'


metric_path = os.path.join(r'C:\Users\MingjingSun\git\5.7 based on 4.30\DSpodPFL_5.7\Exp_data', '47.csv')

# Read the CSV file into a Pandas DataFrame
df = pd.read_csv(metric_path)


plt.figure(figsize=(10, 6))  # Setting figure size

# plot data from a single metric table, e.g., itr and acc; or plot data points from multiple metric tables with different parameters, e.g., convergence speed when num of agents varies
sns.lineplot(data=df, x='iteration', y='rewards', hue='hue_column', ci='sd')  # ci='sd' for confidence intervals

plt.title('Line Plot with Confidence Intervals')  # Adding title
plt.xlabel('Iteration')  # Adding x-axis label
plt.ylabel('Reward')  # Adding y-axis label
plt.legend(title='Hue')  # Adding legend
plt.show() 