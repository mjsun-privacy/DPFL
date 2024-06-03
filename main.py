import os
import random
from datetime import datetime

from src.methods.DPFL.DPFL import DPFL
from src.methods.DSpodFL.DSpodFL import DSpodFL
from src.methods.PureLocal.PureLocal import PureLocal

import gymnasium as gym
import torch
from stable_baselines3 import PPO       # PPO result outperforms A2C
from stable_baselines3.common.env_checker import check_env
import matplotlib.pyplot as plt
# from stable_baselines.common.callbacks import EvalCallback   # evaluate RL model during training periodically and save the best one
# from stable_baselines3.common.env_util import make_vec_env

import sys
import argparse
import argparse as args
import pandas as pd

# run on gpu

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_default_tensor_type(torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor)

def main(row_number):  
    # Load parameters from the pandas DataFrame
    # read the column of Parameters (e.g., num of agents, time slot) in Pandas Data Frame as x axis, and get metric e.g., val loss as y axis, 
    # and pass the parameter and seed here to run main.py. 
    # Save each result to e.g., 47.csv, then plot it by seaborn or matplot. In this way, we can save time if fault arises
    # simid = int(sys.argv[1])      
    # print("First argument:", simid)
    # folder_path = r'C:\Users\MingjingSun\git\5.7 based on 4.30\DSpodPFL_5.7\Exp_data'

    # Load parameters from the pandas DataFrame
    current_dir = os.path.dirname(os.path.abspath(__file__))
    exp_data_dir = os.path.join(current_dir, 'Exp_data')
    exp_path = os.path.join(exp_data_dir, 'exp_df.csv')
    
    # Create a folder to save the results
    current_time = datetime.now().strftime('%m-%d_%H-%M')
    data_dir = os.path.join(exp_data_dir, 'data', current_time)
    os.makedirs(data_dir, exist_ok=True)

    params_df = pd.read_csv(exp_path)
    params = params_df.iloc[row_number - 2].to_dict()    #  access first row of params by 2

    Method_name = params['Method_name']
    Model_name = params['Model_name']
    Dataset_name = params['Dataset_name']
    Num_agents = params['Num_agents']
    Graph_connectivity = params['Graph_connectivity']
    Labels_per_agent = params['Labels_per_agent']   #  0   #! change it
    alpha = 1          # params['Dirichlet_alpha']
    Partition_name = params['Partition_name']
    Data_size = params['Data_size']
    # Batch_size = params['Batch_size']
    # Learning_rate = params['Learning_rate']
    Seed = params['Seed']
    Batch_size = 16
    Learning_rate = 0.01

    print(f'Running {Method_name} with {Model_name} on {Dataset_name} with {Num_agents} agents')
    if(Method_name == 'DPFL'):
    # instantiate my env
       env = DPFL( 
           model_name = 'CNN',
           dataset_name = 'CIFAR10',
           partition_name = Partition_name,
           num_agents = Num_agents,
           graph_connectivity = Graph_connectivity,
           labels_per_agent = Labels_per_agent,
           Dirichlet_alpha = alpha,
           data_size = Data_size,
           batch_size = Batch_size,
           learning_rate = Learning_rate,
           prob_aggr_type = 'full',
           prob_sgd_type = 'full',
           sim_type = 'data_dist',
           prob_dist_params = (0.5, 0.5),    # (alpha, beta) or (min, max)
           termination_delay = 500,
           DandB = (None,1),
           max_episode_steps = 300,
           seed = Seed
           )      #* max_episode_steps = n_steps in PPO()

       print(Labels_per_agent)
    # It will check your custom environment and output additional warnings if needed
       check_env(env)
    # vec_env = make_vec_env(env, n_envs = 4)
    # Callback during training
    # eval_callback = EvalCallback(env...)  record logs during RL training
    # Create a PPO model
       RL = PPO("MlpPolicy", env, verbose = 1, n_steps = 300)    # 500不够还在下降中 -1.10，max num of step() in an episode, regardless of terminate state of a episode
    # Train the model
    # total_timesteps is total number of step(), where n_steps of step() as a episode, after every n_steps calls reset() regardless of terminate state
       RL.learn(total_timesteps = 9000, progress_bar=True)     # 5000
    # Save the model
       RL.save("RLmodel_saved") 
       del RL  # delete trained model to demonstrate loading
    # Load the trained agent
       RL = PPO.load("RLmodel_saved", env=env)
    # Evaluate the model
       vec_env = RL.get_env()                     # sb3 and gym have different interface, here must use vec_env of sb3
       obs = vec_env.reset()                      # clear model
       accs = []

       for i in range (1000):                       # num of step()    # 1000
           mixing_matrix, _state = RL.predict(obs, deterministic = True)
           obs, reward, terminated, info = vec_env.step(mixing_matrix)    # in sb3.step, only 4 output, but newest gym has 5, not env.step
           # record acc
           accs.extend(info[0]['test_acc'])
               
    # record all metrics based on a row of parameters in one table
       
       metric_df = pd.DataFrame({
        'iteration': range(len(accs)),
        'test_acc': accs})
       metric_df.to_csv(os.path.join(data_dir, f"{row_number}_{Method_name}_{Model_name}_{Dataset_name}_{Partition_name}_{Seed}.csv"), index=False)


    elif(Method_name == 'DSpodFL'):
         exp = DSpodFL(
                 model_name= 'CNN',
                 dataset_name= 'CIFAR10',
                 partition_name = Partition_name,   
                 num_epochs= 10,
                 num_agents= Num_agents,
                 graph_connectivity= Graph_connectivity,     # should note this param in other algs
                 labels_per_agent= Labels_per_agent,
                 Dirichlet_alpha= alpha,
                 data_size = Data_size,
                 batch_size= Batch_size,
                 learning_rate= Learning_rate,
                 prob_aggr_type= 'full',
                 prob_sgd_type= 'full',
                 sim_type= 'data_dist',
                 prob_dist_params= (0.5, 0.5),
                 termination_delay= 500,
                 DandB= (None,1),
                 seed= Seed)
         exp.run()
         metric_df = pd.DataFrame({
        'iteration': range(len(exp.accuracies)),
        'test_acc': exp.accuracies})
         metric_df.to_csv(os.path.join(data_dir, f"{row_number}_{Method_name}_{Model_name}_{Dataset_name}_{Partition_name}_{Seed}.csv"), index=False)
        

    elif(Method_name  == 'PureLocal'):
        exp = PureLocal(
                model_name= 'CNN',
                dataset_name= 'CIFAR10',
                partition_name = Partition_name,
                num_epochs= 10,
                num_agents= Num_agents,
                graph_connectivity= Graph_connectivity,     # should note this param in other algs
                labels_per_agent= Labels_per_agent,
                Dirichlet_alpha= alpha,
                data_size = Data_size,
                batch_size= 16,
                learning_rate= Learning_rate,
                seed= Seed)
        exp.reset()
        exp.run()
        metric_df = pd.DataFrame({
        'iteration': range(len(exp.accuracies)),
        'test_acc': exp.accuracies})
        metric_df.to_csv(os.path.join(data_dir, f"{row_number}_{Method_name}_{Model_name}_{Dataset_name}_{Partition_name}_{Seed}.csv"), index=False)
      
    else:
        print("Method not found!")
        

# Debug
""" row_number = 1
method_name = 'DPFL'
main(row_number, method_name) """



if __name__ == '__main__':

    # pass parameters to experiment from Pandasparameter DataFrame
    # Read parameters from the DataFrame
    parser = argparse.ArgumentParser()
    parser.add_argument('row_number', type=int, help="Row number from parameter DataFrame")
    """ parser.add_argument('Method_name', type=str, help="choose a method to run: DPFL, DSpodFL, PureLocal") 
    parser.add_argument('NN_name', type=str, help="choose a NN model: SVM, CNN, VGG11") 
    parser.add_argument('Dataset_name', type=str, help="choose a dataset: MNIST, FMNIST, FEMINIST, CIFAR10") 
    parser.add_argument('Partition_name', type=str, help="choose partition style: by_labels, Dirichlet")  """
    args = parser.parse_args()
    main(args.row_number)



    
















