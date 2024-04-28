import os
import random

from DSpodFL import DSpodFL
import gymnasium as gym
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_default_tensor_type(torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor)


def main():
    env = DSpodFL(
        model_name = 'SVM',
        dataset_name = 'FMNIST',
        num_epochs = 4,
        num_agents = 10,
        graph_connectivity = 0.4,
        labels_per_agent = 1,
        batch_size = 16,
        learning_rate = 0.01,
        prob_aggr_type = 'full',
        prob_sgd_type = 'full',
        sim_type = 'data_dist',
        prob_dist_params = (0.5, 0.5),    # (alpha, beta) or (min, max)
        termination_delay = 500,
        DandB = (None,1)
    )

    env.reset()
    
    log = env.run()

if __name__ == '__main__':
    main()

# In case we also want to register it
# gym.register(
#     id='DSpodFL-v0',
#     entry_point='__main__:DSpodFL',
#     kwargs={'param1': 10, 'param2': 'example', 'param3': [1, 2, 3]},
# )
# env = gym.make('DSpodFL-v0')