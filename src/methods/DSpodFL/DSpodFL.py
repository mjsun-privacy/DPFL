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
from src.methods.DSpodFL.Agent_DSpodFL import Agent_DSpodFL as Agent


class DSpodFL:
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
                 prob_aggr_type: str,
                 prob_sgd_type: str,
                 sim_type: str,
                 prob_dist_params,
                 termination_delay: float,
                 DandB,
                 seed: int):
        
        self.model_name = model_name
        self.num_epochs = num_epochs
        self.num_agents = num_agents
        self.graph_connectivity = graph_connectivity
        self.labels_per_agent = labels_per_agent
        self.Dirichlet_alpha = Dirichlet_alpha
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.prob_aggr_type = prob_aggr_type
        self.prob_sgd_type = prob_sgd_type
        self.sim_type = sim_type
        self.prob_dist_params = prob_dist_params
        self.termination_delay = termination_delay
        self.partition_name = partition_name
        self.data_size = data_size

        self.num_classes, transform, self.num_channels = utils.aux_info(dataset_name, model_name)
        self.train_set, self.global_val_set, self.input_dim = utils.dataset_info(dataset_name, transform)
        print(f"train={len(self.train_set)}, test={len(self.global_val_set)}")

        self.seed = seed  
        np.random.seed(self.seed)
        self.graph = self.generate_graph()
        self.initial_prob_sgds = self.generate_prob_sgds()
        self.prob_aggrs = self.initial_prob_aggrs = self.generate_prob_aggrs(is_initial=True)
        train_sets, val_sets, test_sets = utils.generate_train_val_test_sets(self.train_set, self.num_agents, self.num_classes, self.data_size, self.labels_per_agent, 
                                                             self.Dirichlet_alpha, self.partition_name)
        models, criterion, self.model_dim = self.generate_models()
        self.accuracies = []

        self.agents = self.generate_agents(self.initial_prob_sgds, models, criterion, train_sets, val_sets, test_sets)
        self.DandB = utils.determine_DandB(DandB, self.initial_prob_sgds, self.initial_prob_aggrs)
        print(self.DandB)

    def generate_graph(self):
        while True:
            graph = nx.random_geometric_graph(self.num_agents, self.graph_connectivity)
            if nx.is_k_edge_connected(graph, 1):
                break
        # Debug: everytime call this method, plot it
        nx.draw(graph, with_labels=True)    # neteorkx graph has itself label 0,1..., see relabel method
        plt.show()
        return graph

    def generate_prob_sgds(self):
        if self.prob_sgd_type == 'random':
            return [uniform(self.prob_dist_params[0], self.prob_dist_params[1]) for _ in range(self.num_agents)]
        elif self.prob_sgd_type == 'beta':
            return [betavariate(self.prob_dist_params[0], self.prob_dist_params[1]) for _ in range(self.num_agents)]
        elif self.prob_sgd_type == 'full':
            return [1 for _ in range(self.num_agents)]
        elif self.prob_sgd_type == 'zero':
            return [0 for _ in range(self.num_agents)]

    def generate_prob_aggrs(self, is_initial=False):
        prob_aggrs = [[None for _ in range(self.num_agents)] for _ in range(self.num_agents)]
        for i in range(self.num_agents):
            for j in range(self.num_agents):
                # if j in self.graph.adj[i] and prob_aggrs[i][j] is None:
                if prob_aggrs[i][j] is None:
                    if is_initial:
                        if self.prob_aggr_type == 'random':
                            prob_aggrs[i][j] = prob_aggrs[j][i] = uniform(self.prob_dist_params[0], self.prob_dist_params[1])
                        elif self.prob_aggr_type == 'beta':
                            prob_aggrs[i][j] = prob_aggrs[j][i] = betavariate(self.prob_dist_params[0], self.prob_dist_params[1])
                    elif self.prob_aggr_type in ['random', 'beta']:
                        prob_aggrs[i][j] = prob_aggrs[j][i] = self.initial_prob_aggrs[i][j]
                    elif self.prob_aggr_type == 'full':
                        prob_aggrs[i][j] = prob_aggrs[j][i] = 1
                    elif self.prob_aggr_type == 'zero':
                        prob_aggrs[i][j] = prob_aggrs[j][i] = 0
        return prob_aggrs

    def generate_models(self):
        models, criterion, model_dim = [], None, None
        for _ in range(self.num_agents):
            model, criterion, model_dim = utils.model_info(self.model_name, self.input_dim,
                                                           self.num_classes, self.num_channels)
            models.append(model)
        models = [models[0] for _ in models]
        return models, criterion, model_dim

    def generate_agents(self, prob_sgds, models, criterion, train_sets, val_sets, test_sets):
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
                prob_sgd=prob_sgds[i]
            )
            agents.append(agent_i)

        for i in range(self.num_agents):
            for j in list(self.graph.adj[i]):
                agents[i].add_neighbor(agents[j], self.prob_aggrs[i][j], self.initial_prob_aggrs[i][j])
        return agents

    def run(self):
        # num_iters = 2  # comment this line (this was used for testing)
        num_iters = len(self.train_set) // self.num_agents
        total_iter = 0

        iters, iters_sampled = [], []
        losses = []
      
        for k in range(self.num_epochs):
            print(f"epoch: {k}")

            for i in range(num_iters):
                total_iter = k * num_iters + i
                # print(f"epoch: {k}, iter: {i}, total_iter={total_iter}")
                loss = 0.0
                test_acc = 0.0
                val_loss = 0.0
                cpu_used, max_cpu_usable = 0, 0
                bandwidth_used, max_bandwidth_usable = 0, 0
                transmission_time_used, max_transmission_time_usable = 0, 0
                processing_time_used, max_processing_time_usable = 0, 0
                delay_used, max_delay_usable = 0, 0

                for j in range(self.num_agents):
                    self.agents[j].run_step1()

                test_accs = [0.0]*self.num_agents
                val_losses = [0.0]*self.num_agents

                for j in range(self.num_agents):
                    test_acc += float(self.agents[j].calculate_accuracy())   # float (64)
                    test_accs[j] = self.agents[j].calculate_accuracy() # TODO: verify that across several step(), the Agent remain at the same position in the vector self.agents # Yes, no effect
                    val_loss +=float(self.agents[j].calculate_val_loss()) 
                    val_losses[j] = self.agents[j].calculate_val_loss() 


                    cpu_used += self.agents[j].cpu_used()
                    max_cpu_usable += self.agents[j].max_cpu_usable()
                    processing_time_used += self.agents[j].processing_time_used()
                    max_processing_time_usable += self.agents[j].max_processing_time_usable()
                    bandwidth_used += self.agents[j].bandwidth_used()
                    max_bandwidth_usable += self.agents[j].max_bandwidth_usable()
                    transmission_time_used += self.agents[j].transmission_time_used()
                    # max_transmission_time_usable += self.agents[j].max_transmission_time_usable()

                    #delay_used += self.agents[j].delay_used()
                    #max_delay_usable += self.agents[j].max_delay_usable()

                for j in range(self.num_agents):
                    self.agents[j].run_step2()

                iters.append(total_iter)
                self.accuracies.append(test_acc / self.num_agents)
                print(test_acc / self.num_agents)
        log1 = {"iters": iters,
                "test_accuracy": self.accuracies,}

        return log1


    def reset(self, graph_connectivity=0.4, labels_per_agent=None, prob_aggr_type='full', prob_sgd_type='full',
              sim_type='eff', prob_dist_params=(0, 1), num_agents=10):
        if graph_connectivity != self.graph_connectivity:
            self.reset_graph(graph_connectivity)
        if labels_per_agent != self.labels_per_agent:
            self.reset_train_val_test_sets(labels_per_agent)
        if prob_aggr_type != self.prob_aggr_type:
            self.reset_prob_aggrs(prob_aggr_type)
        if prob_sgd_type != self.prob_sgd_type:
            self.reset_prob_sgds(prob_sgd_type)
        if sim_type != self.sim_type:
            self.reset_sim_type(sim_type)
        if prob_dist_params != self.prob_dist_params:
            self.reset_prob_dist_params(prob_dist_params)
        for agent in self.agents:
            # model, _, _ = utils.model_info(self.model_name, self.input_dim, self.num_classes, self.num_channels)
            # agent.reset(model=model)
            agent.reset()

    def reset_graph(self, graph_connectivity):
        self.graph_connectivity = graph_connectivity
        self.graph = self.generate_graph()

        for i in range(self.num_agents):
            self.agents[i].clear_neighbors()
            for j in list(self.graph.adj[i]):
                self.agents[i].add_neighbor(self.agents[j], self.prob_aggrs[i][j], self.initial_prob_aggrs[i][j])

    def reset_train_val_test_sets(self, labels_per_agent):
        self.labels_per_agent = labels_per_agent
        train_sets, val_sets, test_sets = utils.generate_train_val_test_sets(self.train_set, self.num_agents, self.num_classes, self.labels_per_agent, 
                                                             self.Dirichlet_alpha, self.partition_name)
        for i in range(self.num_agents):
            self.agents[i].set_train_set(train_sets[i])
            self.agents[i].set_val_set(val_sets[i])
            self.agents[i].set_test_set(test_sets[i])

    def reset_prob_aggrs(self, prob_aggr_type):
        self.prob_aggr_type = prob_aggr_type
        self.prob_aggrs = self.generate_prob_aggrs()

        for i in range(self.num_agents):
            for j in list(self.graph.adj[i]):
                self.agents[i].set_prob_aggr(self.agents[j], self.prob_aggrs[i][j])

    def reset_prob_sgds(self, prob_sgd_type):
        self.prob_sgd_type = prob_sgd_type
        prob_sgds = self.generate_prob_sgds()

        for i in range(self.num_agents):
            self.agents[i].set_prob_sgd(prob_sgds[i])

    def reset_sim_type(self, sim_type):
        self.sim_type = sim_type

    def reset_prob_dist_params(self, prob_dist_params):
        self.prob_dist_params = prob_dist_params
        self.initial_prob_sgds = self.generate_prob_sgds(is_initial=True)
        self.prob_aggrs = self.initial_prob_aggrs = self.generate_prob_aggrs(is_initial=True)

        for i in range(self.num_agents):
            self.agents[i].reset(prob_sgd=self.initial_prob_sgds[i])
            for j in list(self.graph.adj[i]):
                self.agents[i].set_prob_aggr(self.agents[j], self.prob_aggrs[i][j])
                self.agents[i].set_initial_prob_aggr(self.agents[j], self.initial_prob_aggrs[i][j])

    