import torch
import torch.nn as nn
import torch.distributions.constraints as constraints

import pyro
import pyro.infer
import pyro.optim
import pyro.distributions as dist

from samplednn.utils import ExportUtils, PerfUtils

import time
import os


class FullyConnected(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim, use_cuda=False):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, output_dim)
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        if use_cuda:
            self.cuda()

        self.use_cuda = use_cuda

    def forward(self, x):
        output = self.fc1(x)
        output = nn.functional.relu(output)
        output = self.out(output)
        return output


class BayesianNetwork(ExportUtils, PerfUtils, FullyConnected):

    short_name = 'bayesian'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


    def model(self, features, targets):

        fc1w_prior = dist.Normal(
            loc=torch.zeros_like(self.fc1.weight),
            scale=torch.ones_like(self.fc1.weight),
        )
        fc1b_prior = dist.Normal(
            loc=torch.zeros_like(self.fc1.bias),
            scale=torch.ones_like(self.fc1.bias),
        )

        outw_prior = dist.Normal(
            loc=torch.zeros_like(self.out.weight),
            scale=torch.ones_like(self.out.weight),
        )
        outb_prior = dist.Normal(
            loc=torch.zeros_like(self.out.bias),
            scale=torch.ones_like(self.out.bias),
        )

        priors = {
            'fc1.weight': fc1w_prior,
            'fc1.bias': fc1b_prior,
            'out.weight': outw_prior,
            'out.bias': outb_prior
        }

        # lift module parameters to random variables sampled from the priors
        lifted_model = pyro.random_module('module', self, priors)
        # sample a regressor (which also samples w and b)
        lifted_reg_model = lifted_model()

        lhat = lifted_reg_model(features)

        pyro.sample('obs', dist.Categorical(logits=lhat), obs=targets)

    def guide(self, features, targets):

        # First layer weight distribution priors
        fc1w_mean = torch.randn_like(self.fc1.weight)
        fc1w_std = torch.abs(torch.randn_like(self.fc1.weight))
        fc1w_mean_param = pyro.param('fc1w_mean', fc1w_mean)
        fc1w_std_param = pyro.param(
            'fc1w_std', fc1w_std, constraint=constraints.positive)
        fc1w_prior = dist.Normal(
            loc=fc1w_mean_param,
            scale=fc1w_std_param
        )

        # First layer bias distribution priors
        fc1b_mean = torch.randn_like(self.fc1.bias)
        fc1b_std = torch.abs(torch.randn_like(self.fc1.bias))
        fc1b_mean_param = pyro.param('fc1b_mean', fc1b_mean)
        fc1b_std_param = pyro.param(
            'fc1b_std', fc1b_std, constraint=constraints.positive)
        fc1b_prior = dist.Normal(
            loc=fc1b_mean_param,
            scale=fc1b_std_param
        )

        # Output layer weight distribution priors
        outw_mean = torch.randn_like(self.out.weight)
        outw_std = torch.abs(torch.randn_like(self.out.weight))
        outw_mean_param = pyro.param('outw_mean', outw_mean)
        outw_std_param = pyro.param(
            'outw_std', outw_std, constraint=constraints.positive)
        outw_prior = dist.Normal(
            loc=outw_mean_param,
            scale=outw_std_param
        )

        # Output layer bias distribution priors
        outb_mean = torch.randn_like(self.out.bias)
        outb_std = torch.abs(torch.randn_like(self.out.bias))
        outb_mean_param = pyro.param('outb_mean', outb_mean)
        outb_std_param = pyro.param(
            'outb_std', outb_std, constraint=constraints.positive)
        outb_prior = dist.Normal(
            loc=outb_mean_param,
            scale=outb_std_param
        )

        priors = {
            'fc1.weight': fc1w_prior,
            'fc1.bias': fc1b_prior,
            'out.weight': outw_prior,
            'out.bias': outb_prior
        }

        lifted_module = pyro.random_module('module', self, priors)

        return lifted_module()


    def save(self, filename):
        pyro.get_param_store().save(filename)

    def load(self, filename):
        pyro.get_param_store().load(filename)


    def train(self, dataloader, num_epochs=15, optimizer=pyro.optim.Adam,
              learning_rate=0.01, loss_function=pyro.infer.Trace_ELBO(),
              print_time=False, print_loss=True, export_params=False,
              export_name=None):

        if export_params:
            export_name, params_dir, index_dir = self.setup_folder_structure(export_name)
            train_info = {
                'num_epochs': num_epochs,
                'optimizer': optimizer.__name__,
                'learning_rate': learning_rate,
                'loss_function': loss_function
            }
            self.save_index(dataloader, train_info, export_name, index_dir)

        start_time = time.time()
        batch_step = len(dataloader) // 15

        optim = optimizer({'lr': learning_rate})
        svi = pyro.infer.SVI(self.model, self.guide, optim, loss=loss_function)

        for epoch in range(num_epochs):
            loss = 0
            for batch, data in enumerate(dataloader):
                features, targets = data
                features = features.view(-1, self.input_dim)

                if self.use_cuda:
                    features = features.cuda()
                    targets = targets.cuda()

                # calculate the loss and take a gradient step
                loss += svi.step(features, targets)

                if print_loss and batch % batch_step == 0:
                    print('.', end='')

            normalized_loss = loss / len(dataloader.dataset)

            if print_loss:
                print('\nEpoch {}\tLoss: {}'.format(epoch, normalized_loss))

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

            outputs = []
            for _ in range(num_samples):
                sampled_guide = self.guide(None, None)
                outputs.append(sampled_guide(features.view(-1,self.input_dim)))
        
        return torch.stack(outputs).transpose(0,1)

    def predict_all(self, dataloader, num_samples):

        with torch.no_grad():
            sampled_guides = [self.guide(None, None) for _ in range(num_samples)]
            
            outputs = []
            for batch, data in enumerate(dataloader):

                print('Predicting batch {} of {}'.format(
                    batch, len(dataloader)
                ), end='\r', flush=True)

                features, targets = data

                if self.use_cuda:
                    features = features.cuda()
                
                predictions = [g(features.view(-1, self.input_dim)) for g in sampled_guides]
                predictions = torch.stack(predictions)
                predictions = predictions.transpose(0,1)
                
                outputs.append(predictions)

        return torch.cat(outputs)