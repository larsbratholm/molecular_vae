import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import joblib

from main import VAE, Model


class RepresentationDataset(Dataset):
    """ FCHL representation dataset
    """

    def __init__(self, idx=None, nuclear_charge=1):
        self.features = self._get_features(idx, nuclear_charge)

    def _get_features(self, idx, nuclear_charge):
        """ Extract all representations of atype from
            molecules with indices idx
        """
        representations = []
        nuclear_charges = joblib.load("ml/charges.pkl")
        training_representations = joblib.load("ml/representations.pkl")

        if idx is None:
            idx = range(len(training_representations))

        for i in idx:
            for j in range(len(nuclear_charges[i])):
                if nuclear_charges[i][j] == nuclear_charge:
                    representations.append(training_representations[i,j])
        representation_array = np.asarray(representations)
        # mean and scale (shouldn't be needed)
        representation_array -= np.mean(representation_array)
        representation_array /= np.std(representation_array)
        return representation_array.astype(np.float32)

    def __len__(self):
        return self.features.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = self.features[idx]
        return sample, sample


if __name__ == "__main__":
    dataset = RepresentationDataset(nuclear_charge=1)
    vae_structure = VAE(dataset.features.shape[-1], layer_size=128, n_layers=2, variant=3,
        dimensions=2, activation=F.leaky_relu)
    model = Model(dataset=dataset, model=vae_structure, epochs=50, learning_rate=3e-3, batch_size=100,
        log_interval=100)
    print(3e-2)
    model.fit()

    model.set_learning_rate(1e-2)
    print(1e-2)
    model.epochs = 30
    model.fit()

    model.set_learning_rate(3e-3)
    print(3e-3)
    model.epochs = 90
    model.fit()

    model.set_learning_rate(1e-3)
    print(1e-3)
    model.epochs = 30
    model.fit()

    model.set_learning_rate(3e-4)
    print(3e-4)
    model.epochs = 20
    model.fit()

