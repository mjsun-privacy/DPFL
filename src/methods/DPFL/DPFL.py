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
from src.methods.DPFL.Agent_my import Agent_DPFL as Agent



def expand_matrix(matrix):
    # Get the number of rows (n) and columns (n-1) of the input matrix
    n, n_minus_1 = matrix.shape
    
    # Create an n x n matrix filled with zeros
    expanded_matrix = np.zeros((n, n))
    
    # Insert the elements from the input matrix into the new matrix
    for i in range(n):
        expanded_matrix[i, :i] = matrix[i, :i]
        expanded_matrix[i, i+1:] = matrix[i, i:]
    
    return expanded_matrix




class DPFL(gym.Env):      # my_env, subclass of class gym.Env  (not a wrapper)
    def __init__(self,
                 model_name: str,
                 dataset_name: str,
                 partition_name: str,
                 num_agents: int,
                 graph_connectivity: float,
                 labels_per_agent: int,
                 Dirichlet_alpha: float,
                 data_size: float,
                 batch_size: int,
                 learning_rate: float,
                 max_episode_steps: int,
                 seed: int):

        self.model_name = model_name
        self.num_agents = num_agents
        self.graph_connectivity = graph_connectivity
        self.labels_per_agent = labels_per_agent
        self.Dirichlet_alpha = Dirichlet_alpha
        self.batch_size = batch_size
        self.learning_rate = learning_rate


        self.partition_name = partition_name
        self.data_size = data_size

        #Call constructor of parent class and specify action and state spaces
        # inherit attributes action_space and obs_space from Class gym.env and redefine     # mj: in multi rl, env, action and state spaces should be in Class Agents
        super(DPFL, self).__init__()
        #self.action_space = spaces.Box(low=0.0, high=1.0, shape=(self.num_agents, self.num_agents - 1), dtype=np.float64) # flattened mixing matrix  #* should not be flattened
        discrete_values = np.array([0.0, 0.5, 1.0])
        self.action_space = spaces.MultiDiscrete([len(discrete_values)] * (self.num_agents * (self.num_agents - 1)))
        self.observation_space = spaces.Box(low=0.0, high=100, shape=(self.num_agents,), dtype=np.float64) #TODO: these are the losses. find how to give no high bound  #self.num_agents,   

        # self.obs = [0]*self.num_agents # New: contains the system state (observation). Initialized at zero
        
        self.num_classes, transform, self.num_channels = utils.aux_info(dataset_name, model_name)
        
        self.train_set, self.global_val_set, self.input_dim = utils.dataset_info(dataset_name, transform)
        print(f"train={len(self.train_set)}, val={len(self.global_val_set)}")

        self.seed = seed  
        np.random.seed(self.seed)
        self.graph = self.generate_graph()                                              # changes when calling self.generate_graph
    

        train_sets, val_sets, test_sets = utils.generate_train_val_test_sets(self.train_set, self.num_agents, self.num_classes, self.data_size, self.labels_per_agent, 
                                                             self.Dirichlet_alpha, self.partition_name)
        models, criterion, self.model_dim = self.generate_models()

        self.agents = self.generate_agents(models, criterion, train_sets, val_sets, test_sets)
    
        
        # trigger truncated = True and call reset() accoring to tutorial of sb3
        self._max_episode_steps = max_episode_steps
        self._elapsed_steps = None
        self.training_records = {'action': [], 'reward': [], 'test_acc': []}
        

    # subclass adds new methods   
    #* current code is fine
    #* rgg only determines who communicates with who, e.g., if in graph, node 0 indexed by graph and node 1 by graph is connected, 
    #* then Agent[id=0] communicates with Agent[id=1] and access its id correctly, and Agent[id = 1] its test_data will not be messed up by others, always together 
    # generate_graph is only called in the beginning as self.graph and called in reset() as the start of each episode (terminate when RL achieves its goal)
    #* never called during step 1 and 2.  self.graph.adj[i] is only to access property, determine neighbors, not calling generate_graph
    # we don't need continous rgg, so the generated rgg can be arbitrary
    def generate_graph(self):
        while True:
            graph = nx.random_geometric_graph(self.num_agents, self.graph_connectivity)
            if nx.is_k_edge_connected(graph, 1):
                break
        # Debug: everytime call this method, plot it
        nx.draw(graph, with_labels=True)    # neteorkx graph has itself label 0,1..., see relabel method
        plt.show()
        return graph
    
    # always generate same graph, self.seed increase until find a connected graph
    def generate_fixed_graph(self):
        while True:            
            # Create the graph with specified node labels
            graph = nx.random_geometric_graph(self.num_agents, self.graph_connectivity, seed=self.seed)
            # Ensure the graph is connected
            if nx.is_k_edge_connected(graph, 1):
                return graph
            else:
                self.seed = self.seed + 1


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
        #* Agents atrributes are static, their id attribute and test_data attribute are corresponded, will not be messed up
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
            print(f"Agent {i} has {len(train_sets[i])} training samples, {len(val_sets[i])} val samples, {len(test_sets[i])} test samples")
            agents.append(agent_i)

        
        for i in range(self.num_agents):
            #* determine who communicate with who according to generated rgg.
            #* not calling rgg, just accessing agj property of existing rgg
            print(len(list(self.graph.adj[i])))
            for j in list(self.graph.adj[i]):
                agents[i].add_neighbor(agents[j])
        return agents


    # reconstruct step method in parent class
    #* Should follow the standard: one parameter: mixing_matrix, sb3 and gym will know it as action
    # Should keep original run here: once receiving a mixing_matrix, runs only on DL, not on RL. 
    #* change graph only when a step() is done, as the start of next step, never change graph in the iteration loop in step()!  
    def step(self, mixing_matrix):    
        #* each sb3 step contains a rgg, certain sgd steps and aggr.

        # TimeLimit wrapper required by sb3 (wrapper: without changing source code) for each episode, we intergrate this wrapper in code
        self._elapsed_steps += 1
        truncated = False
        if self._elapsed_steps >= self._max_episode_steps:
           truncated = True
           print('Truncated!')

        info = {}
        
        # Map 0 -> 0, 1 -> 0.5, 2 -> 1
        mixing_matrix = np.reshape(mixing_matrix, (self.num_agents, self.num_agents - 1))
        mixing_matrix = mixing_matrix * 0.5
      
        #mixing_matrix = np.clip(mixing_matrix, 0.0, 1.0)
        # fill diogonal elements 0 for simplicity
        expanded_mixing_matrix = expand_matrix(mixing_matrix)
        print(f"current action is:{expanded_mixing_matrix}")
        
        self.training_records['action'].append(expanded_mixing_matrix)
        
        #mixing_matrix = np.zeros((self.num_agents, self.num_agents))
        #np.fill_diagonal(mixing_matrix, 1)
        # print(f"current unnormalized action is:{mixing_matrix}")     
        # avoid illegal action, NO NEED TO NORMAILIZE, sum = 1 naturally holds, we just have to ensure wij \in [0,1].
        # we can not control how and where the RL generate actions (e.g., illegal), so post-processe: set 0 for non-neighbor and then normalize 1
        # or don't set 0, since non-neighbor of graph will not have communication, so no effect no matter how RL generates
        # check if the generated action satisfies action_space. our action space is already (0,1), we should not use softmax
        # during the itrs, generate_graph will not be called, hence, neighbors will not change 
            # print(f"epoch:{k}")
            # num_iters = len(self.train_set) // self.num_agents, data will not be reused    
            # iters num require for one epoch (passing all training_data once for sgd), we don't need epoch
            # if DL runs multiple steps, will w begins correctly from previous DL round and as the start of next RL round? Yes, self.w records
        
        num_iters = 5          # more sgd steps and aggrs using each action enables RL to learn more, such as to reduce the noise from sgd 
        for i in range(num_iters):

            for j in range(self.num_agents):
                self.agents[j].run_step1(expanded_mixing_matrix)

            for j in range(self.num_agents):
                self.agents[j].run_step2()

        test_acc = 0.0
        val_loss = 0.0
        test_accs = [0.0]*self.num_agents
        val_losses = [0.0]*self.num_agents 
        accuracy_avg = 0.0
        #cpu_used, max_cpu_usable = 0, 0
        #bandwidth_used, max_bandwidth_usable = 0, 0
        #transmission_time_used, max_transmission_time_usable = 0, 0
        #processing_time_used, max_processing_time_usable = 0, 0
        #delay_used, max_delay_usable = 0, 0

                
        for j in range(self.num_agents):
            test_acc += float(self.agents[j].calculate_test_acc())   # float (64)
            test_accs[j] = self.agents[j].calculate_test_acc() # TODO: verify that across several step(), the Agent remain at the same position in the vector self.agents # Yes, no effect
            val_loss += float(self.agents[j].calculate_val_loss())
            val_losses[j] =  self.agents[j].calculate_val_loss()  # python defult to be float 64
           
    


            #cpu_used += self.agents[j].cpu_used()
            #max_cpu_usable += self.agents[j].max_cpu_usable()
            #processing_time_used += self.agents[j].processing_time_used()
            #max_processing_time_usable += self.agents[j].max_processing_time_usable()
            #bandwidth_used += self.agents[j].bandwidth_used()
            #max_bandwidth_usable += self.agents[j].max_bandwidth_usable()
            #transmission_time_used += self.agents[j].transmission_time_used()


        obs = np.array(val_losses, dtype= np.float64)# obs = np.array([0.0]) #
        reward = -val_loss  #2 ** (- val_loss)     # 2 ** (val_acc - 0.89) - 1      #* no, self.loss is training loss #TODO: make sure this is actually the sum of losses on the validation set by checking semantics of self.loss 
        # a RL episode end with terminated state, i.e., achieve final goal:
        terminated = False       #TODO: return true when termination condition is met, e.g. convergence # will call reset(), which resets model and graph
        # An episode is done iff the agent has reached the target, e.g.,
        # terminated = np.array_equal(self._agent_location, self._target_location)
        # reward = 1 if terminated else 0  # Binary sparse rewards
        #terminated = True if val_loss < 0.10 else False
        #if terminated:
        #    print(f"Terminated! val_loss ={val_loss}")
        #    reward += 100       

        # training records       
        self.training_records['reward'].append(reward)
        self.training_records['test_acc'].append(test_acc / self.num_agents)
        
        # eval records
        accuracy_avg = test_acc / self.num_agents
        print(reward, accuracy_avg)
        info['test_acc']= accuracy_avg
        info['test_acc_per_agent'] = test_accs
        return obs, reward, terminated, truncated,  info     # newest gym have added truncated value and info
 
        
  

    # reconstruct reset method in parent class     
    #* reset everything, model, graph....shouldn't be called before completing a episode or before completing PPO n_steps 
    # call reset when terminated = true (a episode done) or sb3 runs given n_steps
    #* when a episode done, terminated return true: calling reset: clear agents' model and graph, everything. 
    #* or when sb3 runs given steps, automatically call this 
    #* now it's fine
    def reset(self, seed= None):
        
        graph_connectivity=self.graph_connectivity
        # labels_per_agent=1
        # prob_aggr_type='full'
        # prob_sgd_type='full'
        # sim_type='data_dist'
        # prob_dist_params=(0.5, 0.5)
        # num_agents=self.num_agents,

        obs = np.array([0.0]*self.num_agents, dtype= np.float64)     # convert list to numpy array   
        info = {}
        
        if graph_connectivity != self.graph_connectivity:
            # regenerate a graph when a episode done if input a new connectivity, otherwise in reset() graph will not change
            self.reset_graph(graph_connectivity)      
        # if labels_per_agent != self.labels_per_agent:
          #  self.reset_train_val_sets(labels_per_agent)
        #if prob_aggr_type != self.prob_aggr_type:
            #self.reset_prob_aggrs(prob_aggr_type)
        #if prob_sgd_type != self.prob_sgd_type:
            #self.reset_prob_sgds(prob_sgd_type)
        #if sim_type != self.sim_type:
            #self.reset_sim_type(sim_type)
        #if prob_dist_params != self.prob_dist_params:
            #self.reset_prob_dist_params(prob_dist_params)

        for agent in self.agents:
            # model, _, _ = utils.model_info(self.model_name, self.input_dim, self.num_classes, self.num_channels)
            # agent.reset(model=model)
            agent.reset()

        self._elapsed_steps = 0
        return obs, info




    """
    def compute_action(self):     #* I think this RL part should be moved to main.py with an instance of myenv: DSpodFL being created
        # TODO: replace the following line with RL agent from SB3, for example action, _state = model.predict(self.obs, deterministic=True)       
        # The problem I see is that if obs is only the losses vector, can the agent understand the task similarity?
        #* on a single graph, it works. For changing graph, we should pass the graph (gym) of each step() to obs.
        mixing_matrix = np.random.rand(self.num_agents,self.num_agents)
        mixing_matrix = (mixing_matrix + np.transpose(mixing_matrix))/2 #just a trick to make it symmetric
   
       
    
        
    def run(self):          #* I think this RL part should be moved to main.py with an instance of myenv: DSpodFL being created
   
       
        total_iter = 0

    

            #* Here moved to main.py
            mixing_matrix = self.compute_action()
            self.obs, reward, done, info = self.step(k,num_iters,mixing_matrix) #this is new: advances the system
            # TODO: save reward in a file or do something with it, e.g., plot it
    """


    def reset_graph(self, graph_connectivity):
        self.graph_connectivity = graph_connectivity
        self.graph = self.generate_graph()

        for i in range(self.num_agents):
            self.agents[i].clear_neighbors()
            for j in list(self.graph.adj[i]):
                self.agents[i].add_neighbor(self.agents[j])

    def reset_train_val_test_sets(self, labels_per_agent):
        self.labels_per_agent = labels_per_agent
        train_sets, val_sets, test_sets = utils.generate_train_val_test_sets(self.train_set, self.num_agents, self.num_classes, self.labels_per_agent, 
                                                             self.Dirichlet_alpha, self.partition_name)
        for i in range(self.num_agents):
            self.agents[i].set_train_set(train_sets[i])
            self.agents[i].set_val_set(val_sets[i])
            self.agents[i].set_test_set(test_sets[i])
             


    def set_global_seed(seed=42):
        random.seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)


 