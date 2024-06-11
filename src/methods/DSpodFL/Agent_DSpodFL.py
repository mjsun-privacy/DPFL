from copy import deepcopy     # copy1
from random import choices, random

import torch
from torch import optim
from torch.utils.data import DataLoader

import utils
import numpy as np


class Agent_DSpodFL:
    def __init__(self,
                 id,
                 initial_model,
                 criterion,
                 train_set,
                 val_set,
                 test_set,
                 batch_size: int,
                 learning_rate: float,
                 ):
        self.id = id,
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
        self.loss = 0

        self.accuracy = 0.0
        self.val_loss = 0.0
        self.data_processed = 0
        self.aggregation_count = 0

    def run_step1(self):
        self.gradient = [0 for _ in range(self.len_params)]
        self.gradient = self.event_data(choices(self.train_set, k=self.batch_size))

        self.aggregation = [0 for _ in range(self.len_params)]
        self.aggregation_neighbors = []
        for neighbor in self.neighbors:
            self.aggregation_neighbors.append(neighbor)
        if len(self.aggregation_neighbors) != 0:
            self.aggregation = self.event_aggregation()

    def run_step2(self):
        with torch.no_grad():
            param_idx = 0
            for param in self.w.parameters():
                param.data += self.aggregation[param_idx] - self.gradient[param_idx]
                param_idx += 1


    def event_data(self, data):
        self.data_processed += self.batch_size
        return self.gradient_descent(data)

    def event_aggregation(self):
        aggregation = [0 for _ in range(self.len_params)]
        for neighbor in self.neighbors:
            aggregation_weight = self.calculate_aggregation_weight(neighbor['agent'])

            param_idx = 0
            for param, param_neighbor in zip(self.w.parameters(), neighbor['agent'].get_w().parameters()):
                aggregation[param_idx] += aggregation_weight * (param_neighbor.data - param.data)
                param_idx += 1

        self.aggregation_count += len(self.neighbors)
        return aggregation

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
        dataX, dataY = next(iter(train_loader))
        dataX, dataY = dataX.to(self.device), dataY.to(self.device)

        optimizer = optim.SGD(w2.parameters(), lr=self.learning_rate)
        optimizer.zero_grad()
        output = w2(dataX)
        loss = self.criterion(output, dataY)
        loss.backward()
        optimizer.step()

        gradient = [None for _ in range(self.len_params)]

        param_idx = 0
        for param, param2 in zip(self.w.parameters(), w2.parameters()):
            gradient[param_idx] = param.data - param2.data
            param_idx += 1

        self.loss = loss
        return gradient

    def calculate_aggregation_weight(self, neighbor_agent):
        return 1 / (1 + max(self.get_degree(), neighbor_agent.get_degree()))
    

    def calculate_val_loss(self):
         self.w.eval()
         val_loader = DataLoader(
                      self.val_set,
                      batch_size=32,
                      shuffle=False
                      )             
         
         val_loss = 0.0
         num_samples = 0
         with torch.no_grad():
              for dataX, dataY in val_loader:     
                  dataX, dataY = dataX.to(self.device), dataY.to(self.device)

                  output = self.w(dataX)
                  batch_loss = self.criterion(output, dataY)
                  val_loss += batch_loss.item() * dataX.size(0)    
                  num_samples += dataX.size(0)

         return val_loss / num_samples      



    def calculate_accuracy(self):
        self.accuracy = utils.calculate_accuracy(self.w, self.test_set)
        return self.accuracy

    def reset(self, model=None):


        # Learning-based parameters
        if model is not None:
            self.w = model  # generate new random weights
        else:
            self.w = deepcopy(self.initial_model)  # reuse initial model every time
        self.loss = 0



    def add_neighbor(self, agent):
        self.neighbors.append({'agent': agent})

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

    def get_loss(self):
        return self.loss

    def get_aggregation_count(self):
        return self.aggregation_count

    def get_aggregation_neighbors_count(self):
        return len(self.aggregation_neighbors)



    def set_train_set(self, train_set):
        self.train_set = train_set

    def set_val_set(self, val_set):
        self.val_set = val_set

    def set_test_set(self, test_set):    
        self.test_set = test_set    
