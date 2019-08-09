import torch
import torch.nn as nn

import pyro
import pyro.infer
import pyro.optim
import pyro.distributions as dist

from samplednn.utils import ExportUtils, PerfUtils

import time
import os


class Decoder(nn.Module):
    def __init__(self, z_dim, hidden_dim, input_dim):
        super().__init__()
        # setup the two linear transformations used
        self.fc1 = nn.Linear(z_dim, hidden_dim)
        self.fc21 = nn.Linear(hidden_dim, input_dim)
        # setup the non-linearities
        self.softplus = nn.Softplus()
        self.sigmoid = nn.Sigmoid()

    def forward(self, z):
        # define the forward computation on the latent z
        # first compute the hidden units
        hidden = self.softplus(self.fc1(z))
        # return the parameter for the output Bernoulli
        # each is of size batch_size x input_dim
        loc_img = self.sigmoid(self.fc21(hidden))
        return loc_img


class Classifier(nn.Module):
    def __init__(self, z_dim, hidden_dim, class_dim):
        super().__init__()
        # setup two linear transformations
        self.fc1 = nn.Linear(z_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, class_dim)
        self.softplus = nn.Softplus()
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, z):
        # define the forward computation on the latent z
        # first compute the hidden units
        hidden = self.softplus(self.fc1(z))
        # return the predicted class scores
        scores = self.softmax(self.fc2(hidden))
        return scores


class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, z_dim):
        super().__init__()
        # setup the three linear transformations used
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc21 = nn.Linear(hidden_dim, z_dim)
        self.fc22 = nn.Linear(hidden_dim, z_dim)
        # setup the non-linearities
        self.softplus = nn.Softplus()

    def forward(self, x):
        # define the forward computation on the image x
        # first shape the mini-batch to have pixels in the rightmost dimension
        x = x.reshape(-1, 784)
        # then compute the hidden units
        hidden = self.softplus(self.fc1(x))
        # then return a mean vector and a (positive) square root covariance
        # each of size batch_size x z_dim
        z_loc = self.fc21(hidden)
        z_scale = torch.exp(self.fc22(hidden))
        return z_loc, z_scale


class VAE(ExportUtils, PerfUtils, nn.Module):

    short_name = 'vae'

    # by default our latent space is 50-dimensional
    # and we use 400 hidden units
    def __init__(self, input_dim, class_dim, z_dim=50, hidden_dim_ae=400, hidden_dim_cl=256, use_cuda=False):
        super().__init__()

        # create the encoder and decoder networks
        self.encoder = Encoder(input_dim, hidden_dim_ae, z_dim)
        self.decoder = Decoder(z_dim, hidden_dim_ae, input_dim)
        self.classifier = Classifier(z_dim, hidden_dim_cl, class_dim)

        if use_cuda:
            # calling cuda() here will put all the parameters of
            # the encoder and decoder networks into gpu memory
            self.cuda()
        self.use_cuda = use_cuda
        self.input_dim = input_dim
        self.class_dim = class_dim
        self.z_dim = z_dim

    # define the model p(x|z)p(z)
    def model(self, x, labels):
        # register PyTorch module `decoder` with Pyro
        pyro.module("decoder", self.decoder)
        pyro.module("classifier", self.classifier)
        with pyro.plate("data", x.shape[0]):
            # setup hyperparameters for prior p(z)
            z_loc = x.new_zeros(torch.Size((x.shape[0], self.z_dim)))
            z_scale = x.new_ones(torch.Size((x.shape[0], self.z_dim)))
            # sample from prior (value will be sampled by guide when computing the ELBO)
            z = pyro.sample("latent", dist.Normal(z_loc, z_scale).to_event(1))
            # decode the latent code z
            loc_img = self.decoder.forward(z)
            scores = self.classifier.forward(z)
            # score against actual images
            pyro.sample("obs_img", dist.Bernoulli(loc_img).to_event(1), obs=x.reshape(-1, self.input_dim))
            # score against actual class label
            pyro.sample("obs_scores", dist.Categorical(logits=scores), obs=labels)

    # define the guide (i.e. variational distribution) q(z|x)
    def guide(self, x, labels):
        # register PyTorch module `encoder` with Pyro
        pyro.module("encoder", self.encoder)
        with pyro.plate("data", x.shape[0]):
            # use the encoder to get the parameters used to define q(z|x)
            z_loc, z_scale = self.encoder.forward(x)
            # sample the latent code z
            pyro.sample("latent", dist.Normal(z_loc, z_scale).to_event(1))

    
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

    # define a helper function for reconstructing images
    def reconstruct(self, x):
        # encode image x
        z_loc, z_scale = self.encoder(x)
        # sample in latent space
        z = dist.Normal(z_loc, z_scale).sample()
        # decode the image (note we don't sample in image space)
        loc_img = self.decoder(z)
        return loc_img
    
    def classify(self, x):
        # endocde image x
        z_loc, z_scale =  self.encoder(x)
        # sample in latent space
        z = dist.Normal(z_loc, z_scale).sample()
        # classify the image
        scores = self.classifier(z)
        return scores

    def predict(self, x, num_samples):
        # encode image x
        z_loc, z_scale = self.encoder(x)
        # sample in latent space
        z = dist.Normal(z_loc, z_scale).sample(torch.Size([num_samples]))
        # classify the latent space samples
        return self.classifier(z)

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

                z_loc, z_scale = self.encoder(features.view(-1, self.input_dim))
                
                z = dist.Normal(z_loc, z_scale).sample(torch.Size([num_samples]))

                predictions = self.classifier(z)
                predictions = predictions.transpose(0,1)

                outputs.append(predictions)

        return torch.cat(outputs)

    
    def interpolate(self, x, y, num_steps=10):
        x_loc, x_scale = self.encoder.forward(x.cuda())
        y_loc, y_scale = self.encoder.forward(y.cuda())
        intermed = [vae.decoder.forward(x_loc + i/(num_steps-1) * (y_loc - x_loc)).cpu().detach().view(28,28) for i in range(num_steps)]
        return torch.stack(intermed)
