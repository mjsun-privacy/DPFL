from itertools import product
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

params = {
    'Method_name': ['DPFL', 'DSpodFL', 'PureLocal'],
    'Model_name': ['CNN'],   # SVM is a margin, not a NN model, margin cannot be aggregated. weak and robust to overfitting,
    'Dataset_name': ['CIFAR10'], #'FMNIST' is too simple, 10 samples oversee the whole dataset 
    'Num_agents': [5], 
    'Graph_connectivity': [100],
    'Labels_per_agent': [1, 2, 10],      # product conflicts with Dirichlet_alpha in DataFrame
    #'Dirichlet_alpha': [0.3,  0.5, 0.8],
    'Partition_name': ['by_labels'],    #,     'Dirichlet'
    'Data_size': [0.1, 0.2],
    #'Batch_size': [16, 16, 16, 16 ],
    #'Learning_rate': [0.01, 0.01, 0.01],
    #'Max_episode_steps': [1000, 1000, 1000],
    'Seed': [42]
    }

keys, values = zip(*params.items())

# Generate all combinations
combinations = [dict(zip(keys, v)) for v in product(*values)]
exp_df = pd.DataFrame(combinations, columns=keys)

sort_columns = [col for col in keys if col != 'Method_name'] + ['Method_name']
exp_df.sort_values(by=sort_columns, inplace=True)

# Reset index to have a clean DataFrame
exp_df.reset_index(drop=True, inplace=True)


# Get the directory of the script
script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, 'exp_df.csv')

exp_df.to_csv(file_path, index=False)

print(exp_df)
print(f"Experiment table saved to {file_path}")