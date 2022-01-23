import numpy as np
from torch.utils.data import Dataset
import torch


class TFDataset(Dataset):
    def __init__(self, file_name, scaler=None):
        self.data = np.load(file_name)
        self.x = np.array(self.data['x'])[:, :, :, 0]
        self.y = np.array(self.data['y'])[:, :, :, 0]
        if scaler != None:
            self.x[..., 0] = scaler.transform(self.x[..., 0])
        self.len = self.x.shape[0]

    def __getitem__(self, index):
        # x: [N, T, C] y: [N, T, C]
        candidate = [index % (288 * 7)]
        cur = index % (288 * 7)
        while True:
            cur += 288 * 7
            if cur >= self.len:
                break
            candidate.append(cur)
        positive_index = cur[np.random.randint(0, len(cur))]
        while True:
            negative_index = np.random.randint(0, self.len)
            if negative_index % (288 * 7) != index % (288 * 7):
                break
        return {
            'anchor_x': torch.from_numpy(self.x[index]).permute(1, 0, 2).float(),
            'anchor_y': torch.from_numpy(self.y[index]).permute(1, 0, 2).float(),
            'positive_x': torch.from_numpy(self.x[positive_index]).permute(1, 0, 2).float(),
            'positive_y': torch.from_numpy(self.y[positive_index]).permute(1, 0, 2).float(),
            'negative_x': torch.from_numpy(self.x[negative_index]).permute(1, 0, 2).float(),
            'negative_y': torch.from_numpy(self.y[negative_index]).permute(1, 0, 2).float(),
        }

    def __len__(self):
        return self.len
