import neurogym as ngym
import numpy as np
import matplotlib.pyplot as plt
import ipdb 
import torch
import importlib

class Dataset:
    def __init__(self, env, batch_size, seq_len=400):
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.env = env
        self._mask = None
    
    # Make supervised dataset
        self.dataset = ngym.Dataset(
            env, batch_size=self.batch_size, seq_len=self.seq_len)
        self.ob_size = self.env.observation_space.shape[0]
        self.act_size = self.env.action_space.n

    def plot_env(self):
        fig = ngym.utils.plot_env(self.env, num_trials=3)
        plt.show()

    @property
    def mask(self):
        if self._mask is None:
            inputs, labels = self.dataset()
            n_time, batch_size, _ = inputs.shape
            
            # Identify response period indexes where inputs are all zeros
            zero_mask = np.all(inputs == 0 , axis=2)

            # Identify response period indexes where inputs are all zeros
            mask = np.where(zero_mask, 5, 1)

            # Modify mask for continuous sequences of fives
            for b in range(batch_size):
                count = 0
                for i in range(n_time):
                    if mask[i, b] == 5:
                        if count < 4:
                            mask[i, b] = 2 + count
                        count += 1
                    else:
                        count = 0  # Reinitialize the count when a non-five value is encountered

            self._mask = mask

        return mask




def get_class_instance(class_name, **kwargs):
    module = importlib.import_module('main')
    class_ = getattr(module, class_name)
    instance = class_(**kwargs)
    return instance
    