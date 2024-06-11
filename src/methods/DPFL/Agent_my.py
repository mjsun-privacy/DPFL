from copy import deepcopy     # copy1
from random import choices, random

import torch
from torch import optim
from torch.utils.data import DataLoader

import utils
import numpy as np
import torch.nn.functional as F




class Agent_DPFL:
    def __init__(self,
                 id,
                 initial_model,
                 criterion,
                 train_set,
                 val_set,
                 test_set,
                 batch_size: int,
                 learning_rate: float,
                 ):                       #Multi RL: should have parameter gym env by each agent
        self.id = id
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.initial_model = initial_model
        self.criterion = criterion
        self.train_set = train_set
        self.val_set = val_set
        self.test_set = test_set
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        
        self.w = deepcopy(self.initial_model)
        self.len_params = len(list(initial_model.parameters()))

        self.neighbors = []
        self.aggregation_neighbors = []
        self.gradient = None
        self.aggregation = None
        self.loss = 0 # TODO: understand if this is a training or validation loss # was None  
                      #* self.loss = training loss
        self.accuracy = 0.0
        self.val_acc = 0.0
        self.val_loss = 0.0
        self.data_processed = 0
        self.aggregation_count = 0
       
    def run_step1(self, mixing_matrix):
         
        #* Do gradient first, from 
        # if you're selected to do an sgd step, do it and update the gradient, self.loss etc
        self.gradient = [0 for _ in range(self.len_params)]       
        
        
        self.gradient = self.gradient_descent(choices(self.train_set, k=self.batch_size))

        #vhat decides if j is in i's neighbor list
        #at this point the neighbors list is already full with the correct vhats
        #we determine the aggregation_neighbors as a subset
        self.aggregation_neighbors = []
        for neighbor in self.neighbors:
            self.aggregation_neighbors.append(neighbor)
               
        
        self.aggregation = [0 for _ in range(self.len_params)]
        # if i has neighbors, aggregate their models into i
        if len(self.aggregation_neighbors) != 0:     
            
            
            # original normalization   leads to bad performance, 随着agents增多，没有normalize 会无限发散，有强制sum=1 normalize会up and down，不相关的会有系数
            action_sum = 0.0
            for neighbor in self.aggregation_neighbors: 
                action_sum += mixing_matrix[self.id][neighbor['agent'].id]   


            if action_sum == 0.0:
                pass
            elif action_sum > 1.0: 
               for neighbor in self.aggregation_neighbors:
                   mixing_matrix[self.id][neighbor['agent'].id]  /= (1.0* action_sum)
                   
            



            #mixing_matrix[self.id] = softmax(mixing_matrix[self.id])   #! it's wrong! a same matrix will change if each action has multiple aggr times.
            #mixing_matrix = np.identity(mixing_matrix.shape[0])  
            #print(f"current action of agent{self.id} is:{softmax(mixing_matrix[self.id])}") 
           
           
            # mixing_matrix[1:, :] = 0
            #* wii in mixing_matrix is useless, because in update wii = 1-\sum wij, and sum = 1 always holds. 
            #* To recover L2C, we can let \sum wij \in [0，1], so that wii \in [0,1], 
            for neighbor in self.aggregation_neighbors: 
                param_idx = 0                           
                # neighbor['agent'] will be pointed to a instance of Class Agent(), so it can call method get_w of Class Agent  
                for param, param_neighbor in zip(self.w.parameters(), neighbor['agent'].get_w().parameters()):      
                    #if F.cosine_similarity(param_neighbor.data.flatten(), param.data.flatten(), dim=0)>0:                 # 增加cosine similarity的限制，只有相似的才聚合, 但应该让RL自己学习，不应该人为限制
                       self.aggregation[param_idx] += mixing_matrix[self.id][neighbor['agent'].id] * (param_neighbor.data - param.data)     # 如果模型相差太大不应该聚合，导致发散
                       param_idx += 1

            self.aggregation_count += len(self.aggregation_neighbors)


    def run_step2(self):
        with torch.no_grad():
            param_idx = 0
            for param in self.w.parameters():
                # see def， param is a tuple [str, data]. param.data points to the parameter of w
                param.data += self.aggregation[param_idx] - self.gradient[param_idx]   #* param.data updates self.w directly, hence changed self.w is the start of new iteraion
                param_idx += 1



    def gradient_descent(self, data):
        # We do the update on a temporary model, so that we can do the gradient descent
        # and the aggregation at the same iteration later.
        w2 = deepcopy(self.w).to(self.device)
        w2.train()

        train_loader = DataLoader(
            data,
            batch_size=self.batch_size,
            shuffle=True
        )
        # next(iter()) 每次一个batch不重复, 遍历完一次training set就是一个epoch
        dataX, dataY = next(iter(train_loader))                 # everytime trains on a batch of data of train_loader, so no loop here
        dataX, dataY = dataX.to(self.device), dataY.to(self.device)

        optimizer = optim.SGD(w2.parameters(), lr=self.learning_rate)    # Adam is better than SGD, less noise and faster convergence
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


    def calculate_test_acc(self):                       
        self.accuracy = utils.calculate_accuracy(self.w, self.test_set)
        return self.accuracy


    def calculate_val_acc(self):                       
        self.accuracy = utils.calculate_accuracy(self.w, self.val_set)
        return self.val_acc
   

    def reset(self, model=None):
        # Agent-based properties
       
        # Learning-based parameters
        if model is not None:
            self.w = model  # generate new random weights
        else:
            self.w = deepcopy(self.initial_model)  # reuse initial model every time
        self.loss = 0

    
    def add_neighbor(self, agent):
        self.neighbors.append({'agent': agent})
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

    def set_train_set(self, train_set):
        self.train_set = train_set

    def set_val_set(self, val_set):
        self.val_set = val_set

    def set_test_set(self, test_set):
        self.test_set = test_set


