import torch
from torch.utils.data import Dataset, DataLoader
from lib.utils.tools import read_pkl

class MotionDatasetRep(Dataset):
    def __init__(self, file_path):
        self.rep_data = read_pkl(file_path)

    def __len__(self):
        return len(self.rep_data)

    def __getitem__(self, index):
        data = self.rep_data[index]
        return data['rep'], data['gt'][0]