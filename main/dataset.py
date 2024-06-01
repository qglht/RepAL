import neurogym as ngym
import numpy as np
import matplotlib.pyplot as plt
import ipdb 
import torch
from torch.utils.data import Dataset, DataLoader
import importlib

class CustomDataset(Dataset):
    def __init__(self, env, batch_size, seq_len=400):
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.env = env
        self._mask = None
        self.dataset = ngym.Dataset(
            env, batch_size=self.batch_size, seq_len=self.seq_len)
        self.ob_size = self.env.observation_space.shape[0]
        self.act_size = self.env.action_space.n

    def __len__(self):
        return self.batch_size

    def __getitem__(self, idx):
        inputs, labels = self.dataset()
        mask = self._create_mask(inputs)
        return inputs, labels, mask

    def _create_mask(self, inputs):
        n_time, batch_size, _ = inputs.shape
        zero_mask = np.all(inputs == 0 , axis=2)
        mask = np.where(zero_mask, 5, 1)
        for b in range(batch_size):
            count = 0
            for i in range(n_time):
                if mask[i, b] == 5:
                    if count < 4:
                        mask[i, b] = 2 + count
                    count += 1
                else:
                    count = 0
        return mask




def get_class_instance(class_name, **kwargs):
    module = importlib.import_module('main')
    class_ = getattr(module, class_name)
    instance = class_(**kwargs)
    return instance
    