import torch
import torch.nn as nn
import torch.nn.functional as functional

from samplednn.utils import ExportUtils, PerfUtils

import time
import os


class NNWithDropout(ExportUtils, PerfUtils, nn.Module):

    short_name = 'dropout'

    def __init__(self, input_dim, hidden_dim, output_dim, p=0.0, use_cuda=False):
        super().__init__()
        self.p = p
        self.input_dim = input_dim
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, output_dim)
                
        if use_cuda:
            self.cuda()
            
        self.use_cuda = use_cuda
        
    def forward(self, x):
        output = self.fc1(x)
        output = functional.relu(functional.dropout(output, p=self.p, training=True))
        output = self.out(output)
        return output

    
    def save(self, filename):
        torch.save(self.state_dict(), filename)


    def load(self, filename):
        self.load_state_dict(torch.load(filename))


    def train(self, dataloader, num_epochs=15, optimizer=torch.optim.SGD,
              learning_rate=0.01, loss_function=nn.CrossEntropyLoss(),
              print_time=False, print_loss=True, export_params=False,
              export_name=None):

        if export_params:
            export_name, params_dir, index_dir = self.setup_folder_structure(export_name)
            train_info = {
                'num_epochs': num_epochs,
                'optimizer': optimizer.__name__,
                'learning_rate': learning_rate,
                'loss_function': loss_function,
                'dropout_rate': self.p
            }
            self.save_index(dataloader, train_info, export_name, index_dir)

        start_time = time.time()
        batch_step = len(dataloader) // 15

        optimizer = optimizer(self.parameters(), lr=learning_rate)
        criterion = loss_function

        for epoch in range(num_epochs):
            running_loss = 0
            for batch, data in enumerate(dataloader):
                features, targets = data
                features = features.view(-1, self.input_dim)

                if self.use_cuda:
                    features = features.cuda()
                    targets = targets.cuda()
                
                optimizer.zero_grad()

                outputs = self(features)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                if print_loss and batch % batch_step == 0:
                    print('.', end='')

            if print_loss:
                print('\nEpoch {}\tLoss: {}'.format(epoch, running_loss))

            if export_params:
                self.save_params(epoch, export_name, params_dir)

            if print_time:
                print('Time taken since start: {} s'.format(
                    time.time()-start_time))
    
        if print_loss:
            print(('=' * 10) + '\nTraining done!')

    
    def predict(self, x, num_samples):
        
        with torch.no_grad():
            features = x
            if self.use_cuda:
                features = features.cuda()
            
            outputs = [self.forward(features.view(-1, self.input_dim)) for _ in range(num_samples)]
        
        return torch.stack(outputs)


    def predict_all(self, dataloader, num_samples):

        with torch.no_grad():

            outputs = []
            for batch, data in enumerate(dataloader):

                print('Predicting batch {} of {}'.format(
                    batch, len(dataloader)
                ), end='\r', flush=True)

                features, targets = data

                if self.use_cuda:
                    features = features.cuda()

                predictions = [self.forward(features.view(-1, self.input_dim)) for _ in range(num_samples)]
                predictions = torch.stack(predictions)
                predictions = predictions.transpose(0,1)

                outputs.append(predictions)
        
        return torch.cat(outputs)
