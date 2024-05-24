# from itertools import product
import pandas as pd
import os

""" numnodes = [1, 2, 3]
list2 = ['a', 'b']
list3 = ['x', 'y'] 

# Generate Cartesian product
cartesian_product = list(product(numnodes, list2, list3))

# Create DataFrame
df = pd.DataFrame(cartesian_product, columns=['numnodes', 'List2', 'List3'])

print(df)
df.to_csv('exp.csv') """

exp_df = {
    'Model_name': ['SVM', 'SVM', 'SVM'],
    'Dataset_name': ['FMNIST', 'FMNIST', 'FMNIST'],
    'Num_agents': [10, 20, 30], 
    'Graph_connectivity': [0.2, 0.4, 0.6],
    'Labels_per_agent': [1, 2, 3],
    'Dirichlet_alpha': [0.1, 0.5, 0.8],
    'Batch_size': [16, 16, 16],
    'Learning_rate': [0.01, 0.01, 0.01],
    'Max_episode_steps': [1000, 1000, 1000],
    'Seed': [42, 123, 456]
    }


exp_df = pd.DataFrame(exp_df, columns=exp_df.keys())

folder_path = r'C:\Users\MingjingSun\git\5.7 based on 4.30\DSpodPFL_5.7\Exp_data'
file_path = os.path.join(folder_path, 'exp_df.csv')
exp_df.to_csv(file_path, index=False)

print(exp_df)
print(f"Experiment table saved to {file_path}")