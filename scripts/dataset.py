import torch
from torch.utils.data import Dataset
import numpy as np
from tqdm import tqdm

class TimeSeriesDataset(Dataset):
    def __init__(self, data, seq_length, prediction_horizon, scaler=None):

        self.seq_length = seq_length
        self.data = data
        
        if scaler:
            self.data['Global_active_power'] = scaler.fit_transform(
                self.data[['Global_active_power']].values
            )
        self.data = self.data.values
        
        self.X, self.y = self.create_sequences(self.data, seq_length, prediction_horizon)

    def create_sequences(self, data, seq_length, prediction_horizon):
        X, y = [], []
        for i in tqdm(range(0, len(data) - seq_length - prediction_horizon), desc="Creating sequences"):
            X.append(data[i:i + seq_length, :]) 
            y.append(data[i + seq_length:i + seq_length + prediction_horizon, :]) 
        return np.array(X), np.array(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx], dtype=torch.float32), torch.tensor(self.y[idx], dtype=torch.float32)