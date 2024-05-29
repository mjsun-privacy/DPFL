from copy import deepcopy     # copy1
from random import choices, random

import torch
from torch import optim
from torch.utils.data import DataLoader

import utils
import numpy as np

class Agent_base:
    # The base class for all agents in the system for any algorithm
    def __init__(self,
                 id,
                 initial_model,
                 criterion,
                 train_set,
                 val_set,
                 batch_size: int,
                 learning_rate: float,
                 ):                       #Multi RL: should have parameter gym env by each agent
        self.id = id
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.initial_model = initial_model
        self.criterion = criterion
        self.train_set = train_set
        self.val_set = val_set
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
        self.val_loss = 0.0
        self.data_processed = None
        self.aggregation_count = None

    
    def run_step1(self):
        self.gradient = [0 for _ in range(self.len_params)]    #* can remove choices, directly DataLoader
        self.gradient = self.gradient_descent(choices(self.train_set, k = self.batch_size))
        with torch.no_grad():
            param_idx = 0
            for param in self.w.parameters():
                # see def， param is a tuple [str, data]. param.data points to the parameter of w
                param.data = param.data - self.gradient[param_idx]   #* param.data updates self.w directly, hence changed self.w is the start of new iteraion
                param_idx += 1


    def gradient_descent(self, data):
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

        self.loss = loss / self.batch_size
        return gradient


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
        self.accuracy = utils.calculate_accuracy(self.w, self.val_set)
        return self.accuracy

   
    def reset(self, model=None):


        if model is not None:
            self.w = model  
        else:
            self.w = deepcopy(self.initial_model)  
        self.loss = 0

        # Counters
        self.data_processed = 0
        self.aggregation_count = 0

    def set_train_set(self, train_set):
        self.train_set = train_set

    def set_val_set(self, val_set):
        self.val_set = val_set