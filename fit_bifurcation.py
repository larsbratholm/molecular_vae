import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import joblib

from main import VAE, Model

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
            return distances.reshape(distances.shape[0], -1).astype(np.float32)
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

if __name__ == "__main__":
    dataset = BifuricationDataset(transform='distances')
    vae_structure = VAE(dataset.features.shape[-1])
    #dataloader = DataLoader(dataset, batch_size=10,
    #                shuffle=True, num_workers=0)
    #data_iter = iter(dataloader)
    #a, b = data_iter.next()

    model = Model(dataset=dataset, model=vae_structure, epochs=20, learning_rate=1e-3)
    model.fit()
