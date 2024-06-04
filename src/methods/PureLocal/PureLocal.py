from random import uniform, betavariate
import numpy as np
import networkx as nx
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt
import random
import torch
import numpy as np
import os

import utils
from src.methods.PureLocal.Agent_base import Agent_base as Agent

class PureLocal:
     def __init__(self,
                 model_name: str,
                 dataset_name: str,
                 partition_name: str,
                 num_epochs: int,
                 num_agents: int,
                 graph_connectivity: float,
                 labels_per_agent: int,
                 Dirichlet_alpha: float,
                 data_size: float,
                 batch_size: int,
                 learning_rate: float,
                 seed: int):
        
        self.model_name = model_name
        self.num_epochs = num_epochs
        self.num_agents = num_agents
        self.graph_connectivity = graph_connectivity
        self.labels_per_agent = labels_per_agent
        self.Dirichlet_alpha = Dirichlet_alpha
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.partition_name = partition_name
        self.data_size = data_size

        self.num_classes, transform, self.num_channels = utils.aux_info(dataset_name, model_name)
        self.train_set, self.global_val_set, self.input_dim = utils.dataset_info(dataset_name, transform)
        print(f"train={len(self.train_set)}, test={len(self.global_val_set)}")

        self.seed = seed  
        np.random.seed(self.seed)
        train_sets, val_sets, test_sets = utils.generate_train_val_test_sets(self.train_set, self.num_agents, self.num_classes, self.data_size, self.labels_per_agent, 
                                                             self.Dirichlet_alpha, self.partition_name)
        models, criterion, self.model_dim = self.generate_models()

        self.agents = self.generate_agents(models, criterion, train_sets, val_sets, test_sets)
        self.accuracies = []


     def generate_models(self):
         models, criterion, model_dim = [], None, None
         for _ in range(self.num_agents):
             model, criterion, model_dim = utils.model_info(self.model_name, self.input_dim,
                                                           self.num_classes, self.num_channels)
             models.append(model)
         models = [models[0] for _ in models]
         return models, criterion, model_dim
     
     
     def generate_agents(self, models, criterion, train_sets, val_sets, test_sets):
         agents = []
         for i in range(self.num_agents):
            agent_i = Agent(
                id=i,
                initial_model=models[i],
                criterion=criterion,
                train_set=train_sets[i],
                val_set=val_sets[i],
                test_set=test_sets[i],
                batch_size=self.batch_size,
                learning_rate=self.learning_rate,
                )
            agents.append(agent_i)
         return agents   


     def run(self):
        # num_iters = 2  # comment this line (this was used for testing)
        #num_iters = len(self.train_set) // self.num_agents
        total_iter = 0

        iters, iters_sampled = [], []
        losses= []

        #for k in range(self.num_epochs):
            #print(f"epoch: {k}")

        for i in range(300):
                #total_iter = k * num_iters + i
                # print(f"epoch: {k}, iter: {i}, total_iter={total_iter}")
                loss = 0.0
                test_acc = 0.0
                val_loss = 0.0

                for j in range(self.num_agents):
                    self.agents[j].run_step1()

                test_accs = [0.0]*self.num_agents
                val_losses = [0.0]*self.num_agents

                for j in range(self.num_agents):
                    test_acc += float(self.agents[j].calculate_accuracy())   # float (64)
                    test_accs[j] = self.agents[j].calculate_accuracy() # TODO: verify that across several step(), the Agent remain at the same position in the vector self.agents # Yes, no effect
                    #val_loss +=float(self.agents[j].calculate_val_loss()) 
                    #val_losses[j] = self.agents[j].calculate_val_loss() 


                iters.append(total_iter)
                self.accuracies.append(test_acc / self.num_agents)
                print(test_acc / self.num_agents)

        log1 = {"iters": iters,
                "test_accuracy": self.accuracies,}

        return log1
     


     def reset(self):
        # Do we need this?
        #if labels_per_agent != self.labels_per_agent:
            #self.reset_train_val_test_sets(labels_per_agent)
     

        for agent in self.agents:
            # model, _, _ = utils.model_info(self.model_name, self.input_dim, self.num_classes, self.num_channels)
            # agent.reset(model=model)
            agent.reset()


     def reset_train_val_test_sets(self, labels_per_agent):
        self.labels_per_agent = labels_per_agent
        train_sets, val_sets, test_sets = utils.generate_train_val_test_sets(self.train_set, self.num_agents, self.num_classes, self.labels_per_agent, 
                                                             self.Dirichlet_alpha, self.partition_name)
        for i in range(self.num_agents):
            self.agents[i].set_train_set(train_sets[i])
            self.agents[i].set_val_set(val_sets[i])
            self.agents[i].set_test_set(test_sets[i])
