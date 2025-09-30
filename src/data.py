import numpy as np
import torch
from torch.utils.data import Dataset

class DriftDataset(Dataset):
    """A dataset class for concept drift adaptation."""

    def __init__(self, data: np.ndarray, target: np.ndarray, x_drift: np.ndarray):
        """
        Initialize the DriftDataset.

        Parameters:
        - data: Input data samples as a numpy array.
        - target: Corresponding labels for the input data.
        - x_drift: drift data without label.
        """
        self.data = np.array(data)
        self.target = np.array(target)
        self.x_drift = np.array(x_drift)

    def __getitem__(self, index: int) -> tuple:
        """
        Retrieve a sample and its ramdom drift sample.

        Parameters:
        - index: The index of the sample to retrieve.

        Returns:
        - A tuple containing the original sample, a drift sample, and the target label.
        """
        random_idx = np.random.randint(0, len(self.x_drift))
        drift_sample = torch.tensor(self.x_drift[random_idx], dtype=torch.float32)
        sample = torch.tensor(self.data[index], dtype=torch.float32)
        return sample, drift_sample, self.target[index]

    def __len__(self) -> int:
        """Return the total number of samples in the dataset."""
        return len(self.data)

    @property
    def shape(self) -> tuple:
        """Return the shape of the input data."""
        return self.data.shape