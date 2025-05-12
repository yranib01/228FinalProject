from load_dataset import *
import torch

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __getitem__(self, idx):
        sample = {
            "features": torch.tensor(self.x[idx], dtype=torch.float),
            "labels": torch.tensor(self.y[idx], dtype=torch.float)
        }

    def __len__(self):
        return len(self.x)


my_dataset = CustomDataset(split, ys)

train_data, val_data = torch.utils.data.random_split(my_dataset, [0.8, 0.2])



