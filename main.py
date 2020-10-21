"""
VAE encoding of molecular representations.
Modified code from https://github.com/pytorch/examples/blob/master/vae/main.py
"""



from __future__ import print_function
#import argparse
import torch
#import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
#from torchvision import datasets, transforms
#from torchvision.utils import save_image
from torch.utils.data import Dataset, DataLoader
import glob
import numpy as np

from filereaders import XYZReader


#parser = argparse.ArgumentParser(description='VAE MNIST Example')
#parser.add_argument('--batch-size', type=int, default=128, metavar='N',
#                    help='input batch size for training (default: 128)')
#parser.add_argument('--epochs', type=int, default=10, metavar='N',
#                    help='number of epochs to train (default: 10)')
#parser.add_argument('--no-cuda', action='store_true', default=False,
#                    help='disables CUDA training')
#parser.add_argument('--seed', type=int, default=1, metavar='S',
#                    help='random seed (default: 1)')
#parser.add_argument('--log-interval', type=int, default=10, metavar='N',
#                    help='how many batches to wait before logging training status')
#args = parser.parse_args()
#args.cuda = not args.no_cuda and torch.cuda.is_available()
#
#torch.manual_seed(args.seed)
#
#
#kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
#train_loader = torch.utils.data.DataLoader(
#    datasets.MNIST('../data', train=True, download=True,
#                   transform=transforms.ToTensor()),
#    batch_size=args.batch_size, shuffle=True, **kwargs)
#test_loader = torch.utils.data.DataLoader(
#    datasets.MNIST('../data', train=False, transform=transforms.ToTensor()),
#    batch_size=args.batch_size, shuffle=True, **kwargs)
#

class BifuricationDataset(Dataset):
    """ Bifurication dataset
    """

    def __init__(self, transform=None):
        filenames = glob.glob("./bifurication/*.xyz")
        self.reader = XYZReader(filenames)
        self.coordinates = self.reader.coordinates
        self.transform = transform
        self.features = self._transform()

    def _transform(self):
        if self.transform == "distances":
            distances = np.linalg.norm(self.coordinates[:,:,None] - self.coordinates[:,None,:], axis=3)
            # ravel
            return distances.reshape(distances.shape[0], 1, -1).astype(np.float32)
        else:
            return self.coordinates

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        #sample = {'feature': self.features[idx]}
        sample = self.features[idx]
        return sample, sample

class VAE(nn.Module):
    def __init__(self, input_size):
        super(VAE, self).__init__()
        self.input_size = input_size
        self.fc1 = nn.Linear(self.input_size, 128)
        self.fc21 = nn.Linear(128, 2)
        self.fc22 = nn.Linear(128, 2)
        self.fc3 = nn.Linear(2, 128)
        self.fc4 = nn.Linear(128, self.input_size)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        #h1 = self.fc1(x)
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        #h3 = self.fc3(z)
        return F.softplus(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, self.input_size))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


class Model(object):
    def __init__(self, epochs=10, batch_size=20, log_interval=1000):
        filenames = glob.glob('*.xyz')
        reader = XYZReader(filenames)

        dataset = BifuricationDataset(transform='distances')
        self.dataloader = DataLoader(dataset, batch_size=batch_size,
                        shuffle=True, num_workers=0)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = VAE(dataset.features.shape[1]).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=3e-3)
        self.epochs = epochs
        self.log_interval = log_interval

    def loss_function(self, recon_x, x, mu, logvar):
        """ Reconstruction + KL divergence losses summed over all elements and batch
        """
        MSE = F.mse_loss(recon_x, x.view(-1, self.model.input_size), reduction='sum')

        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return MSE + KLD


    def fit(self):
        for epoch in range(1, self.epochs + 1):
            self.model.train()
            train_loss = 0
            for batch_idx, (data, _) in enumerate(self.dataloader):
                data = data.to(self.device)
                self.optimizer.zero_grad()
                recon_batch, mu, logvar = self.model(data)
                loss = self.loss_function(recon_batch, data, mu, logvar)
                loss.backward()
                train_loss += loss.item()
                self.optimizer.step()
                if (batch_idx+1) % self.log_interval == 0:
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch, batch_idx * len(data), len(dataloader.dataset),
                        100. * batch_idx / len(dataloader),
                        loss.item() / len(data)))

            print('====> Epoch: {} Average loss: {:.4f}'.format(
                  epoch, train_loss / len(dataloader.dataset)))

    #def predict(self, 
    #        with torch.no_grad():
    #            sample = torch.randn(64, 20).to(self.device)
    #            sample = model.decode(sample).cpu()
    #            save_image(sample.view(64, 1, 28, 28),
    #                       'results/sample_' + str(epoch) + '.png')


#TODO optimize activations
#TODO exponential lr decay
#TODO try layer/weight norm
#TODO Optimize layer size
#TODO normalize input
#TODO inverse distance matrix / distance embedding / fchl
#TODO distances to coordinates reconstruction
if __name__ == "__main__":
    filenames = glob.glob('*.xyz')
    reader = XYZReader(filenames)

    dataset = BifuricationDataset(transform='distances')
    dataloader = DataLoader(dataset, batch_size=10,
                    shuffle=True, num_workers=0)
    data_iter = iter(dataloader)
    a, b = data_iter.next()

    model = Model(epochs=50)
    model.fit()



