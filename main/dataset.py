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

# class DatasetSingleton:
#     _instances = {}

#     @classmethod
#     def get_instance(cls, env, batch_size, seq_len, hp, **kwargs):
#         key = (env, batch_size, seq_len, frozenset(kwargs.items()))
#         if key not in cls._instances:
#             env_instance = get_class_instance(env, config=hp)
#             cls._instances[key] = ngym.Dataset(env_instance, batch_size=batch_size, seq_len=seq_len)
#         return cls._instances[key]
    
def swap_axes(inputs, targets):
    return np.swapaxes(inputs, 0, 1), np.swapaxes(targets, 0, 1)
    
class NeuroGymDataset(Dataset):
    def __init__(self, env, batch_size, device, hp, seq_len, num_pregenerated=512):
        self.num_pregenerated = num_pregenerated
        self.num_samples = int(hp["max_steps"])
        self.device = device
        self.env = env
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.hp = hp
        self.dataset = None
        self._generate_data()

    def gen_feed_data(self, inputs, labels):
        batch_size, n_time = inputs.shape[:2]

        new_shape = (batch_size, n_time, self.hp["rule_start"] + self.hp["n_rule"])  
        x = np.zeros(new_shape, dtype=np.float32)  # np.zeros instead of torch.zeros
        ind_rule = self.hp["rules"].index(self.env)
        x[:, :, :self.hp["rule_start"]] = inputs  # Slicing works the same
        x[:, :, self.hp["rule_start"] + ind_rule] = 1  # Setting values
        inputs = x

        return inputs, labels
    
    def _generate_data(self):
        env_instance = get_class_instance(self.env, config=self.hp)
        dataset = ngym.Dataset(env_instance, batch_size=32, seq_len=self.seq_len)
        inputs, targets = np.empty((self.num_pregenerated, self.seq_len, self.hp["n_input"])), np.empty((self.num_pregenerated, self.seq_len))
        for i in range(self.num_pregenerated//32):
            input_sample, target_sample = dataset()
            input_sample, target_sample = swap_axes(input_sample, target_sample)
            input_sample, target_sample = self.gen_feed_data(input_sample, target_sample)
            inputs[i*32:(i+1)*32] = input_sample
            targets[i*32:(i+1)*32] = target_sample
        input_sample, target_sample = dataset()
        input_sample, target_sample = swap_axes(input_sample, target_sample)
        input_sample, target_sample = self.gen_feed_data(input_sample, target_sample)
        inputs[self.num_pregenerated//32*32:] = input_sample[:self.num_pregenerated%32]
        targets[self.num_pregenerated//32*32:] = target_sample[:self.num_pregenerated%32]
        masks = self._create_mask(inputs)
        # convert inputs, targets, masks to torch tensors
        inputs, targets, masks = torch.tensor(inputs).to(torch.float32), torch.tensor(targets).to(torch.float32), torch.tensor(masks).to(torch.float32)
        self.dataset = (inputs, targets, masks) if self.dataset is None else (torch.cat((self.dataset[0], inputs), dim=0), torch.cat((self.dataset[1], targets), dim=0), torch.cat((self.dataset[2], masks), dim=0))

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        try:
            return self.dataset[0][idx,:,:], self.dataset[1][idx,:], self.dataset[2][idx,:]
        except:
            self._generate_data()
            return self.__getitem__(idx)

    def _create_mask(self, inputs):
        n_sample, n_time, _ = inputs.shape
        zero_mask = np.all(inputs == 0 , axis=2)
        mask = np.where(zero_mask, 5.0, 1.0).astype(np.float32)
        for b in range(n_sample):
            count = 0
            for i in range(n_time):
                if mask[b,i] == 5:
                    if count < 4:
                        mask[b,i] = 2 + count
                    count += 1
                else:
                    count = 0
        return mask
    

# def worker_init_fn(worker_id):
#     np.random.seed(np.random.get_state()[1][0] + worker_id)

def get_dataloader(env, batch_size, device, num_workers, hp, shuffle, seq_len = 400):
    dataset = NeuroGymDataset(env, batch_size, device, hp, seq_len=seq_len)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)# worker_init_fn=worker_init_fn)
    return dataloader

    