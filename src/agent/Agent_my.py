from copy import deepcopy     # copy1
from random import choices, random

import torch
from torch import optim
from torch.utils.data import DataLoader

import utils
import numpy as np

class Agent_DPFL:
    def __init__(self,
                 id,
                 initial_model,
                 criterion,
                 train_set,
                 val_set,
                 batch_size: int,
                 learning_rate: float,
                 prob_sgd: float,
                 ):                       #Multi RL: should have parameter gym env by each agent
        self.id = id
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.initial_model = initial_model
        self.criterion = criterion
        self.train_set = train_set
        self.val_set = val_set
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.prob_sgd = self.initial_prob_sgd = prob_sgd

        self.w = deepcopy(self.initial_model)
        self.len_params = len(list(initial_model.parameters()))

        self.neighbors = []
        self.aggregation_neighbors = []
        self.gradient = None
        self.aggregation = None
        self.loss = 0 # TODO: understand if this is a training or validation loss # was None  
                      #* self.loss = training loss
        self.accuracy = 0.0
        self.val_loss = 0.0
        self.data_processed = None
        self.aggregation_count = None
        self.v = 0 #By default I assume you do not do SGD

    def run_step1(self, mixing_matrix):
         
        #* Do gradient first, from 
        # if you're selected to do an sgd step, do it and update the gradient, self.loss etc
        self.gradient = [0 for _ in range(self.len_params)]       
        if self.v == 1:
            self.data_processed += self.batch_size
            self.gradient = self.gradient_descent(choices(self.train_set, k=self.batch_size))

        #vhat decides if j is in i's neighbor list
        #at this point the neighbors list is already full with the correct vhats
        #we determine the aggregation_neighbors as a subset
        self.aggregation_neighbors = []
        for neighbor in self.neighbors:
            if neighbor['v_hat'] == 1:
                self.aggregation_neighbors.append(neighbor)
                #we reset vhat for later
                neighbor['v_hat'] = 0

        self.aggregation = [0 for _ in range(self.len_params)]
        # if i has neighbors, aggregate their models into i
        if len(self.aggregation_neighbors) != 0:     
            
            #* no need to Normalize the action array, sum = 1 always holds. but should ensure that wij \in [0,1], see MTL, L2C, and this paper. 
            # Here we use wii to affect \sum wij in aggregation update
            action_sum = mixing_matrix[self.id][self.id]
            check_sum = 0.0
            for neighbor in self.aggregation_neighbors: 
                action_sum += mixing_matrix[self.id][neighbor['agent'].id]   

            if action_sum == 1.0:
                check_sum = 1.0
            
            # if sum wij/i == 0: no aggr
            if action_sum == 0.0:       
               mixing_matrix[self.id][self.id] = 1.0
               action_sum = 1.0
               check_sum = 1.0
            
            if action_sum != 1.0:  
               mixing_matrix[self.id][self.id] /= (1.0* action_sum)
               check_sum = mixing_matrix[self.id][self.id]
               for neighbor in self.aggregation_neighbors:
                   mixing_matrix[self.id][neighbor['agent'].id]  /= (1.0* action_sum)
                   check_sum += mixing_matrix[self.id][neighbor['agent'].id]

            # print(f"current action of agent{self.id} is:{mixing_matrix}") 
            # print(f"neighbors summed weights of agent{self.id} is:{check_sum}")

            #* wii in mixing_matrix is useless, because in update wii = 1-\sum wij, and sum = 1 always holds. 
            #* To recover L2C, we can let \sum wij \in [0，1], so that wii \in [0,1], 
            for neighbor in self.aggregation_neighbors: 
                param_idx = 0                           
                # neighbor['agent'] will be pointed to a instance of Class Agent(), so it can call method get_w of Class Agent  
                for param, param_neighbor in zip(self.w.parameters(), neighbor['agent'].get_w().parameters()):                    
                    self.aggregation[param_idx] += mixing_matrix[self.id][neighbor['agent'].id] * (param_neighbor.data - param.data)     # w = copy.deepcopy(model)
                    param_idx += 1

            self.aggregation_count += len(self.aggregation_neighbors)


    def run_step2(self):
        with torch.no_grad():
            param_idx = 0
            for param in self.w.parameters():
                # see def， param is a tuple [str, data]. param.data points to the parameter of w
                param.data += self.aggregation[param_idx] - self.gradient[param_idx]   #* param.data updates self.w directly, hence changed self.w is the start of new iteraion
                param_idx += 1

        self.v = 0
        if random() <= self.prob_sgd:
            self.v = 1

        for neighbor in self.neighbors:
            if random() <= neighbor['prob_aggr']:
                neighbor['v_hat'] = 1
                neighbor['agent'].set_v_hat(self, 1)

    def gradient_descent(self, data):
        # We do the update on a temporary model, so that we can do the gradient descent
        # and the aggregation at the same iteration later.
        w2 = deepcopy(self.w).to(self.device)
        w2.train()

        train_loader = DataLoader(
            data,
            batch_size=self.batch_size,
            shuffle=False
        )
        # next(iter()) 每次一个batch不重复, 遍历完一次training set就是一个epoch
        dataX, dataY = next(iter(train_loader))                 # everytime trains on a batch of data of train_loader, so no loop here
        dataX, dataY = dataX.to(self.device), dataY.to(self.device)

        optimizer = optim.SGD(w2.parameters(), lr=self.learning_rate)
        optimizer.zero_grad()
        output = w2(dataX)
        loss = self.criterion(output, dataY)      # training loss on current training mini-batch, e.g., entropy loss
        loss.backward()
        optimizer.step()

        gradient = [None for _ in range(self.len_params)]

        param_idx = 0
        for param, param2 in zip(self.w.parameters(), w2.parameters()):
            gradient[param_idx] = param.data - param2.data
            param_idx += 1

        self.loss = loss / self.batch_size
        return gradient

  #  def calculate_aggregation_weight(self, neighbor_agent):
     #   return 1 / (1 + max(self.get_degree(), neighbor_agent.get_degree()))

    # if we use the whole fixed val set rather than random test set, rl performance will be better. 
    def calculate_val_loss(self):
         self.w.eval()
         val_loader = DataLoader(
                      self.val_set,
                      batch_size=32,
                      shuffle=False
                      )              # val_set contains multiple batches，Dataloader每次一个batch直到val_set遍历一遍
         
         val_loss = 0.0
         num_samples = 0
         with torch.no_grad():
              for dataX, dataY in val_loader:      # num of loop = len(val_set)/batch_size 因此要用加号, dataX,Y 是每个batch的数据，要把val_loader的数据loop一遍
                  dataX, dataY = dataX.to(self.device), dataY.to(self.device)

                  output = self.w(dataX)
                  batch_loss = self.criterion(output, dataY)
                  val_loss += batch_loss.item() * dataX.size(0)    # batch_loss = avg loss, dataX.size(0) = num of data points in the batch = batch_size
                  num_samples += dataX.size(0)

         return val_loss / num_samples      # or len(self.val_set)


    def calculate_accuracy(self):                       
        self.accuracy = utils.calculate_accuracy(self.w, self.val_set)
        return self.accuracy

   
    def reset(self, model=None, prob_sgd=None):
        # Agent-based properties
        if prob_sgd is not None:
            self.prob_sgd = prob_sgd

        # Learning-based parameters
        if model is not None:
            self.w = model  # generate new random weights
        else:
            self.w = deepcopy(self.initial_model)  # reuse initial model every time
        self.loss = 0

        # Aggregation-based parameters
        self.v = 1

        # Counters
        self.data_processed = 0
        self.aggregation_count = 0

    def cpu_used(self):
        return self.v

    @staticmethod
    def max_cpu_usable():
        return 1

    def processing_time_used(self):
        return self.cpu_used() / self.initial_prob_sgd

    def max_processing_time_usable(self):
        return self.max_cpu_usable() / self.initial_prob_sgd

    def bandwidth_used(self):
        return len(self.aggregation_neighbors)

    def max_bandwidth_usable(self):
        return self.get_degree()

    def transmission_time_used(self):
        transmission_time_used = 0
        for neighbor in self.aggregation_neighbors:
            transmission_time_used += 1 / neighbor['initial_prob_aggr']
        return transmission_time_used / self.get_degree()
        # return transmission_time_used / len(self.aggregation_neighbors)

    def max_transmission_time_usable(self):
        transmission_time_used = 0
        for neighbor in self.neighbors:
            transmission_time_used += 1 / neighbor['initial_prob_aggr']
        return transmission_time_used / self.get_degree()

    def delay_used(self):
        return self.processing_time_used() + self.transmission_time_used()

    def max_delay_usable(self):
        return self.max_processing_time_usable() + self.max_transmission_time_usable()
    
    def add_neighbor(self, agent, prob_aggr, initial_prob_aggr):
        self.neighbors.append({'agent': agent, 'prob_aggr': 1.0,
                               'initial_prob_aggr': 1.0, 'v_hat': 0})
        # self.neighbors.append({'agent': agent, 'prob_aggr': prob_aggr,
        #                 'initial_prob_aggr': initial_prob_aggr, 'v_hat': 0})

    def clear_neighbors(self):
        self.neighbors = []

    def find_neighbor(self, neighbor_agent):
        for neighbor in self.neighbors:
            if neighbor_agent is neighbor['agent']:
                return neighbor
        return None

    def get_degree(self):
        return len(self.neighbors)

    def get_w(self):
        return self.w
     

    def get_aggregation_count(self):
        return self.aggregation_count

    def get_aggregation_neighbors_count(self):
        return len(self.aggregation_neighbors)

    def get_data_processed(self):
        return self.data_processed

    def set_v_hat(self, neighbor_agent, v_hat):
        self.find_neighbor(neighbor_agent)['v_hat'] = v_hat

    def set_train_set(self, train_set):
        self.train_set = train_set

    def set_val_set(self, val_set):
        self.val_set = val_set

    def set_prob_sgd(self, prob_sgd):
        self.prob_sgd = prob_sgd

    def set_prob_aggr(self, neighbor_agent, prob_aggr):
        self.find_neighbor(neighbor_agent)['prob_aggr'] = prob_aggr

    def set_initial_prob_aggr(self, neighbor_agent, initial_prob_aggr):
        self.find_neighbor(neighbor_agent)['initial_prob_aggr'] = initial_prob_aggr
