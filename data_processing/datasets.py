from torch.utils.data import Dataset

class EMGDataset(Dataset):
    """
    PyTorch Dataset for EMG data split into windows, where the first (window_size - 1) time steps
    are used as input and the last time step as the target.
    """
    def __init__(self, inputs, targets):
        self.inputs = inputs
        self.targets = targets
    
    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]