import os
import torch
from torch.utils.data import Dataset, DataLoader
from lib.utils.tools import read_pkl

class MotionDatasetRep(Dataset):
    def __init__(self, folder_path):
        self.data_files = []
        for root, _, files in os.walk(folder_path):
            for f in files:
                if f.endswith(('.pkl', '.pickle')):
                    file_path = os.path.join(root, f)
                    self.data_files.append(file_path)
        # self.rep_data = read_pkl(file_path)

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, index):
        file_path = self.data_files[index]
        data = read_pkl(file_path)
        # data = self.rep_data[index]
        return data['rep'], data['gt'][0]