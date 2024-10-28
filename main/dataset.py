import torch
from torch.utils.data import Dataset, DataLoader
import pickle


class NeuroGymDataset(Dataset):
    """
    Dataset class for the NeuroGym dataset

    Args:
    env (str): The name of the environment
    mode (str): The mode of the dataset (train or test)

    Attributes:
    env (str): The name of the environment
    dataset (dict): The dataset containing the inputs, targets, and masks
    mode (str): The mode of the dataset (train or test)

    Methods:
    load_data: Load the dataset from the file
    __len__: Return the length of the dataset
    __getitem__: Return the inputs, targets, and masks at
    the given index
    """

    def __init__(self, env, mode):
        self.env = env
        self.dataset = None
        self.mode = mode
        self.load_data()

    def load_data(self):
        """
        Load the dataset from the file
        """
        try:
            with open(f"data/{self.env}_{self.mode}.pkl", "rb") as f:
                self.dataset = pickle.load(f)
        except FileNotFoundError:
            with open(f"../data/{self.env}_{self.mode}.pkl", "rb") as f:
                self.dataset = pickle.load(f)

    def __len__(self):
        return self.dataset["inputs"].shape[0]

    def __getitem__(self, idx):
        return (
            self.dataset["inputs"][idx],
            self.dataset["targets"][idx],
            self.dataset["masks"][idx],
        )


def get_dataloader(
    env, batch_size, num_workers, shuffle, mode="train", train_split=0.8
):
    """
    Get the dataloader for the NeuroGym dataset for the given environment and mode (train or test)

    Args:
    env (str): The name of the environment
    batch_size (int): The batch size
    num_workers (int): The number of workers for the dataloader
    shuffle (bool): Whether to shuffle the dataset
    mode (str): The mode of the dataset (train or test)
    train_split (float): The proportion of the dataset to use for training

    Returns:
    dict: A dictionary containing the train and test dataloaders
    """
    dataset = NeuroGymDataset(env, mode=mode)

    # TODO: Add an argument to be able to be able to generate dataset on the go
    # Determine the size of train and test sets
    total_samples = len(dataset)
    train_size = int(train_split * total_samples)
    test_size = total_samples - train_size

    # Split the dataset into train and test sets
    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, test_size]
    )

    # Create dataloaders for train and test sets
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )

    return {"train": train_dataloader, "test": test_dataloader}
