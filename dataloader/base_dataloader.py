# datasets/base_dataset.py

from torch.utils.data import Dataset

class BaseMultiLabelDataset(Dataset):
    """
    child class must implement:
        _get_image(index) -> tensor C,H,W
        _get_labels(index) -> dict[str, float]  e.g. {"MA":0, "HE":1, ...}

    __getitem__ then standardizes output for model
    """

    def __len__(self):
        raise NotImplementedError

    def _get_image(self, idx):
        raise NotImplementedError

    def _get_labels(self, idx):
        raise NotImplementedError

    def __getitem__(self, idx):
        x  = self._get_image(idx)
        yd = self._get_labels(idx)   # dict[str: int]
        return x, yd
