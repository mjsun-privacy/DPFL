import os
import random

from src.method.DPFL import DPFL
from src.method.DSpodFL import DSpodFL
import gymnasium as gym
import torch
from stable_baselines3 import PPO
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

def main(row_number, method_name):
    
    # Load parameters from the pandas DataFrame
    # read the column of Parameters (e.g., num of agents, time slot) in Pandas Data Frame as x axis, and get metric e.g., val loss as y axis, 
    # and pass the parameter and seed here to run main.py. 
    # Save each result to e.g., 47.csv, then plot it by seaborn or matplot. In this way, we can save time if fault arises
    # simid = int(sys.argv[1])      
    # print("First argument:", simid)
    folder_path = r'C:\Users\MingjingSun\git\5.7 based on 4.30\DSpodPFL_5.7\Exp_data'
    exp_path = os.path.join(folder_path, 'exp_df.csv')
    params_df = pd.read_csv(exp_path)
    params = params_df.iloc[row_number - 1].to_dict()  # Subtract 1 since row numbers start from 1

    Model_name = params['Model_name']
    Dataset_name = params['Dataset_name']
    Num_agents = params['Num_agents']
    Graph_connectivity = params['Graph_connectivity']
    Labels_per_agent = params['Labels_per_agent']
    alpha = params['Dirichlet_alpha']
    Batch_size = params['Batch_size']
    Learning_rate = params['Learning_rate']
    Seed = params['Seed']


    if(method_name == 'DPFL'):
    # instantiate my env
       env = DPFL( 
           model_name = Model_name,
           dataset_name = Dataset_name,
           partition_name = 'by_labels',
           num_epochs = 1,
           num_agents = Num_agents,
           graph_connectivity = Graph_connectivity,
           labels_per_agent = Labels_per_agent,
           Dirichlet_alpha = alpha,
           batch_size = 16,
           learning_rate = 0.01,
           prob_aggr_type = 'full',
           prob_sgd_type = 'full',
           sim_type = 'data_dist',
           prob_dist_params = (0.5, 0.5),    # (alpha, beta) or (min, max)
           termination_delay = 500,
           DandB = (None,1),
           max_episode_steps = 1000,
           seed = Seed
           )      #* max_episode_steps = n_steps in PPO()

       print(Labels_per_agent)
    # It will check your custom environment and output additional warnings if needed
       check_env(env)
    # vec_env = make_vec_env(env, n_envs = 4)
    # Callback during training
    # eval_callback = EvalCallback(env...)  record logs during RL training
    # Create a PPO model
       RL = PPO("MlpPolicy", env, verbose = 1, n_steps = 1200)    # 500不够还在下降中 -1.10，max num of step() in an episode, regardless of terminate state of a episode
    # Train the model
    # total_timesteps is total number of step(), where n_steps of step() as a episode, after every n_steps calls reset() regardless of terminate state
       RL.learn(total_timesteps = 12000, progress_bar=True)     # 10000 
    # Save the model
       RL.save("PPO_saved") 
    #del RL  # delete trained model to demonstrate loading
    # Load the trained agent
    #RL = PPO.load("PPO_saved", env=env)
    # Evaluate the model
       vec_env = RL.get_env()                     # sb3 and gym have different interface, here must use vec_env of sb3
       obs = vec_env.reset()                      # clear model
       rewards = [] 

       for i in range (1000):                       # num of step()    # 1000
           mixing_matrix, _state = RL.predict(obs, deterministic = True)
           obs, reward, terminated, info = vec_env.step(mixing_matrix)    # in sb3.step, only 4 output, but newest gym has 5, not env.step
           rewards.append(reward)
               
    # record all metrics based on a row of parameters in one table
       metric_df = pd.DataFrame({
        'iteration': range(1000),
        'rewards': rewards})
       metric_df.to_csv(os.path.join(folder_path, '{row_number}_{method_name}.csv'), index=False)


    elif(method_name == 'DSpodFL'):
         exp = DSpodFL(
                 model_name= Model_name,
                 dataset_name= Dataset_name,
                 partition_name = 'by_labels',   
                 num_epochs= 10,
                 num_agents= Num_agents,
                 graph_connectivity= Graph_connectivity,     # should note this param in other algs
                 labels_per_agent= Labels_per_agent,
                 Dirichlet_alpha= alpha,
                 batch_size= 16,
                 learning_rate= 0.01,
                 prob_aggr_type= 'full',
                 prob_sgd_type= 'full',
                 sim_type= 'data_dist',
                 prob_dist_params= (0.5, 0.5),
                 termination_delay= 500,
                 DandB= (None,1),
                 seed= Seed)
         exp.run()
        
    elif(method_name  == 'Purelocal'):
        
   
        



if __name__ == '__main__':

    # pass parameters to experiment from Pandasparameter DataFrame
    # Read parameters from the DataFrame
    parser = argparse.ArgumentParser()
    parser.add_argument('row_number', type=int, help="Row number from the parameter table")
    parser.add_argument('method_name', type=str, help="choose a method to run") 
    args = parser.parse_args()
    main(args.row_number)



    




















    # instantiate pararell envs: but in differnet envs, graph will be different.
    # pararell env with sb3, or gym.make with gym to instantiate the single env, gym.vector.env. to instantiate pararell envs.
    """ vec_env = make_vec_env(DSpodFL, n_envs = 4, 
                           env_kwargs=dict(model_name = 'SVM', 
                                       dataset_name = 'FMNIST', 
                                       num_epochs = 1, 
                                       num_agents = 10, 
                                       graph_connectivity = 0.4, 
                                       labels_per_agent = 1, 
                                       batch_size = 16, 
                                       learning_rate = 0.01, 
                                       prob_aggr_type = 'full', 
                                       prob_sgd_type = 'full', 
                                       sim_type = 'data_dist', 
                                       prob_dist_params = (0.5, 0.5), 
                                       termination_delay = 500, 
                                       DandB = (None,1))
                                       )  
    """