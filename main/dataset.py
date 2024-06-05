import torch
from torch.utils.data import Dataset, DataLoader
import pickle

    
class NeuroGymDataset(Dataset):
    def __init__(self, env):
        self.env = env
        self.dataset = None
        self.load_data()
    
    def load_data(self):
        # load the pickle data file
        with open(f'data/{self.env}_data.pkl', 'rb') as f:
            self.dataset = pickle.load(f)
    
    def __len__(self):
        return self.dataset["inputs"].shape[0]

    def __getitem__(self, idx):
        return self.dataset["inputs"][idx], self.dataset["targets"][idx], self.dataset["masks"][idx]

def get_dataloader(env, batch_size, num_workers, shuffle, train_split=0.8):
    dataset = NeuroGymDataset(env)
    
    # Determine the size of train and test sets
    total_samples = len(dataset)
    train_size = int(train_split * total_samples)
    test_size = total_samples - train_size
    
    # Split the dataset into train and test sets
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    
    # Create dataloaders for train and test sets
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    
    return {"train":train_dataloader, "test":test_dataloader}

    