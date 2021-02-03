import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import joblib
import glob
from filereaders import XYZReader

from main import VAE, Model

class BifuricationDataset(Dataset):
    """ Bifurication dataset
    """

    def __init__(self, transform=None, irc_only=False):
        if irc_only:
            filenames = glob.glob("./bifurication/*IRC*.xyz")
        else:
            filenames = sorted(glob.glob("./bifurication/*.xyz"))
        self.reader = XYZReader(filenames)
        self.coordinates = self.reader.coordinates
        self.transform = transform
        self.features = self._transform()
        self.features = self.features

    def _transform(self):
        if self.transform == "distances":
            distances = np.linalg.norm(self.coordinates[:,:,None] - self.coordinates[:,None,:], axis=3)
            indices = np.triu_indices(distances.shape[1], 1)
            # Get upper triangle items for all samples
            upper_triangle = np.vstack([item[indices] for item in distances])
            return upper_triangle.astype(np.float32)
        else:
            return self.coordinates

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = self.features[idx]
        return sample, sample

if __name__ == "__main__":
    dataset = BifuricationDataset(transform='distances', irc_only=False)
    vae_structure = VAE(dataset.features.shape[-1], variant=3, dimensions=2, allow_negative_output=False)

    model = Model(dataset=dataset, model=vae_structure, epochs=250, learning_rate=3e-3, batch_size=100)
    model.fit()
    # only predict IRC
    dataset.features = dataset.features[:266]
    vae_data = model.predict(dataset)
    joblib.dump(vae_data, "vae_data.pkl", compress=("lzma", 9), protocol=-1)
