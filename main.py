"""
VAE encoding of molecular representations.
Modified code from https://github.com/pytorch/examples/blob/master/vae/main.py
"""

from __future__ import print_function
import torch
from torch import nn, optim, tensor
from torch.nn import functional as F
from torch.nn.parameter import Parameter
from torch.utils.data import Dataset, DataLoader
import glob
import numpy as np

from filereaders import XYZReader

class VAE(nn.Module):
    def __init__(self, input_size, layer_size=128, dimensions=2, n_layers=1, activation=F.relu,
            variant=1, allow_negative_output=True):
        """
        Variants include 1) all samples and all features use the same variance
        (are reconstructed with the same accuracy), 2) all features use the same variance,
        but the variance can change from sample to sample, and 3) each sample and each
        feature has different variances.
        Variant 1 and 2 should typically be used for e.g. representations where each feature
        are on the same scale and all features are equally important. Variant 3 should be used
        for input like internal distances, where the scale of each feature is different.
        """
        super(VAE, self).__init__()
        self.input_size = input_size
        self.n_layers = n_layers
        self.dimensions = dimensions
        self.variant = variant
        self.allow_negative_output = allow_negative_output
        self.activation = activation
        self.fc1 = nn.ModuleList([nn.Linear(self.input_size, layer_size)])
        self.fc1.extend([nn.Linear(layer_size, layer_size) for _ in range(n_layers - 1)])
        self.fc21 = nn.Linear(layer_size, dimensions)
        self.fc22 = nn.Linear(layer_size, dimensions)
        self.fc3 = nn.ModuleList([nn.Linear(dimensions, layer_size)])
        self.fc3.extend([nn.Linear(layer_size, layer_size) for _ in range(n_layers - 1)])
        self.fc4 = nn.Linear(layer_size, self.input_size)

        if self.variant == 1:
            self.logvar = Parameter(tensor(0.1))
        elif self.variant == 2:
            self.fc41 = nn.Linear(layer_size, 1)#self.input_size)
        else:
            self.fc41 = nn.Linear(layer_size, self.input_size)

    def encode(self, x):
        h1 = self.activation(self.fc1[0](x))
        h1 = nn.LayerNorm(h1.size()[1:])(h1)
        for i in range(1, self.n_layers):
            h1 = self.activation(self.fc1[i](h1))
            h1 = nn.LayerNorm(h1.size()[1:])(h1)

        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        h3 = self.activation(self.fc3[0](z))
        h3 = nn.LayerNorm(h3.size()[1:])(h3)
        for i in range(1, self.n_layers):
            h3 = self.activation(self.fc3[i](h3))
            h3 = nn.LayerNorm(h3.size()[1:])(h3)
        out = self.fc4(h3)
        if not self.allow_negative_output:
            out = F.softplus(out)
        if self.variant == 1:
            return out, self.logvar
        else:
            return out, self.fc41(h3)

    def forward(self, x):
        mu_z, logvar_z = self.encode(x.view(-1, self.input_size))
        z = self.reparameterize(mu_z, logvar_z)
        mu_x, logvar_x = self.decode(z)
        return mu_x, logvar_x, mu_z, logvar_z

class Model(object):
    def __init__(self, dataset, model=None, epochs=10, batch_size=20, log_interval=1000,
            learning_rate=3e-3):
        self.batch_size = batch_size
        self.dataloader = DataLoader(dataset, batch_size=self.batch_size,
                        shuffle=True, num_workers=0)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        if self.model is None:
            self.model = VAE(dataset.features.shape[1]).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.epochs = epochs
        self.log_interval = log_interval

    def loss_function(self, x, mu_x, logvar_x, mu_z, logvar_z):
        """ Reconstruction + KL divergence losses summed over all elements and batch
        """
        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        KLD = -0.5 * torch.sum(1 + logvar_z - mu_z.pow(2) - logvar_z.exp())

        if self.model.variant == 1:
            # Each dimension and sample has same variance
            MSE = F.mse_loss(mu_x, x, reduction='sum')
            nll = 0.5 * (x.shape[0] * x.shape[1] * logvar_x + MSE / logvar_x.exp())
            # regularization (gamma prior)
            #a, b = 1, 80
            #nll -= (a - 1) * logvar_x - b * logvar_x.exp()
        elif self.model.variant == 2:
            # Each dimension has same variance within a sample
            # but each sample can have different variance
            MSE = torch.sum(F.mse_loss(mu_x, x, reduction='none'), dim=1)[:,None]
            nll = 0.5 * torch.sum(x.shape[1] * logvar_x + MSE / logvar_x.exp())
        else:
            # Each dimension and sample has different variance
            MSE = F.mse_loss(mu_x, x, reduction='none')
            nll = 0.5 * torch.sum(logvar_x + MSE / logvar_x.exp())

        return nll + KLD

    def fit(self):
        for epoch in range(1, self.epochs + 1):
            self.model.train()
            train_loss = 0
            for batch_idx, (data, _) in enumerate(self.dataloader):
                x = data.to(self.device)
                self.optimizer.zero_grad()
                mu_x, logvar_x, mu_z, logvar_z = self.model(x)
                loss = self.loss_function(x, mu_x, logvar_x, mu_z, logvar_z)
                loss.backward()
                train_loss += loss.item()
                self.optimizer.step()
                if (batch_idx+1) % self.log_interval == 0:
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch, batch_idx * len(x), len(self.dataloader.dataset),
                        100. * batch_idx / len(self.dataloader),
                        loss.item() / len(x)))
                    #print("Manifold:", mu_z[0].cpu().detach().numpy())
                    #print("Variance:", logvar_z[0].exp().cpu().detach().numpy())
                    #if self.model.variant == 1:
                    #    print("Accuracy:", logvar_x.exp().cpu().detach().numpy())
                    #elif self.model.variant == 2:
                    #    print("Accuracy:", logvar_x.exp().cpu().detach().numpy().mean())
                    #else:
                    #    print("Accuracy:", logvar_x.exp().cpu().detach().numpy().mean())

            print('====> Epoch: {} Average loss: {:.4f}'.format(
                  epoch, train_loss / len(self.dataloader.dataset)))

    def set_learning_rate(self, lr):
        """ Convenience function for updating learning rate
        """
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def predict(self, dataset):
        """ Returns several of the output parameters of the model.
            First n items are the manifold projections (low dimensional space).
        """
        output = np.zeros((len(dataset), self.model.dimensions * 2))
        dataloader = DataLoader(dataset, batch_size=self.batch_size,
                        shuffle=False, num_workers=0)
        with torch.no_grad():
            for batch_idx, (data, _) in enumerate(dataloader):
                x = data.to(self.device)
                mu_z, logvar_z = self.model.encode(x.view(-1, self.model.input_size))
                mu_x, logvar_x = self.model.decode(mu_z)
                idx = np.arange(batch_idx * self.batch_size, batch_idx * self.batch_size + x.shape[0])
                output[idx, :self.model.dimensions] = mu_z
                output[idx, self.model.dimensions:2*self.model.dimensions] = logvar_z
                #output[idx, -1] = logvar_x.flatten()
        return output
