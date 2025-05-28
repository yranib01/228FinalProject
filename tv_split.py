import torch
print("torch.cuda.is_available():", torch.cuda.is_available())   # True면 GPU 빌드

import numpy as np
import loaddata as ld
# from load_dataset import X_raw, y   # global variables from load_dataset.py


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, x, y):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        sample = {
            "features": self.x[idx],
            "labels": self.y[idx]}
        return sample

X_raw = torch.Tensor(np.load("X_raw.npz")['arr_0']).to(device)
X_deep = torch.Tensor(np.load("X_deep.npz")['arr_0']).to(device)
X_fft_selected = torch.Tensor(np.load("X_fft_selected.npz")['arr_0']).to(device)
y = torch.Tensor(np.load("y_range.npz")['arr_0']).to(device)

dataset_raw = CustomDataset(X_raw, y)
dataset_deep = CustomDataset(X_deep, y)
dataset_fft = CustomDataset(X_fft_selected, y)
# split = torch.Tensor(np.array(split)).to(device)
# ys_interp = torch.Tensor(ys_interp).to(device)
# my_dataset = CustomDataset(split, ys_interp)

N = len(dataset_raw)
train_size = int(0.8 * N)
val_size = N - train_size
train_raw, val_raw = torch.utils.data.random_split(dataset_raw, [train_size, val_size])
train_deep, val_deep = torch.utils.data.random_split(dataset_deep, [train_size, val_size])
train_fft, val_fft = torch.utils.data.random_split(dataset_fft, [train_size, val_size])
# generator=torch.Generator().manual_seed(42)