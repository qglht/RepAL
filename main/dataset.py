import neurogym as ngym
import numpy as np
import matplotlib.pyplot as plt
import ipdb 
from main import AntiPerceptualDecisionMaking, PerceptualDecisionMaking

class Dataset:
    def __init__(self, env, batch_size, seq_len):
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.env = env

    
    # Make supervised dataset
        self.dataset = ngym.Dataset(
            env, batch_size=self.batch_size, seq_len=self.seq_len)
        self.ob_size = self.env.observation_space.shape[0]
        self.act_size = self.env.action_space.n

    def plot_env(self):
        fig = ngym.utils.plot_env(self.env, num_trials=2)
        plt.show()

def get_class_instance(class_name, **kwargs):
    class_ = globals()[class_name]
    instance = class_(**kwargs)
    return instance
        
if __name__ == '__main__':
    envid = "PerceptualDecisionMaking"
    env_kwargs = {'dt': 100}
    env = get_class_instance(envid, **env_kwargs)
    batch_size = 32
    seq_len = 100
    dataset = Dataset(env, batch_size, seq_len)
    ipdb.set_trace()