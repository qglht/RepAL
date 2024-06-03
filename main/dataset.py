import neurogym as ngym
import numpy as np
import matplotlib.pyplot as plt
import ipdb 
import torch
from torch.utils.data import Dataset, DataLoader
import importlib


def get_class_instance(class_name, **kwargs):
    module = importlib.import_module('main')
    class_ = getattr(module, class_name)
    instance = class_(**kwargs)
    return instance

class DatasetSingleton:
    _instances = {}

    @classmethod
    def get_instance(cls, env, batch_size, seq_len, hp, **kwargs):
        key = (env, batch_size, seq_len, frozenset(kwargs.items()))
        if key not in cls._instances:
            env_instance = get_class_instance(env, config=hp)
            cls._instances[key] = ngym.Dataset(env_instance, batch_size=batch_size, seq_len=seq_len)
        return cls._instances[key]

class NeuroGymDataset(Dataset):
    def __init__(self, env, batch_size, device, hp, seq_len=400):
        self.num_samples = int(hp["max_steps"])
        self.device = device
        self.env = env
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.hp = hp
        self.current_inputs, self.current_targets = self._generate_data()
    
    def _generate_data(self):
        dataset = DatasetSingleton.get_instance(self.env, self.batch_size, self.seq_len, self.hp)
        return dataset()
    
    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):

        if idx % self.batch_size == 0:
            self.current_inputs, self.current_targets = self._generate_data()

        batch_idx = idx % self.batch_size
        input_sample = self.current_inputs[:, batch_idx, ...]
        target_sample = self.current_targets[:, batch_idx, ...]
        mask_sample = self._create_mask(input_sample)

        input_sample = torch.as_tensor(input_sample, device=self.device)
        label_sample = torch.as_tensor(target_sample, device=self.device)
        mask_sample = torch.as_tensor(mask_sample, device=self.device)

        return input_sample, label_sample, mask_sample

    def _create_mask(self, inputs):
        n_time, _ = inputs.shape
        zero_mask = np.all(inputs == 0 , axis=1)
        mask = np.where(zero_mask, 5, 1)
        count = 0
        for i in range(n_time):
            if mask[i] == 5:
                if count < 4:
                    mask[i] = 2 + count
                count += 1
            else:
                count = 0
        return mask
    

def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)

def get_dataloader(env, batch_size, device, num_workers, hp):
    dataset = NeuroGymDataset(env, batch_size, device, hp, seq_len=400)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, worker_init_fn=worker_init_fn)
    return dataloader

    