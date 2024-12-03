import torch
from torch.utils.data import Dataset

class EMGDataset(Dataset):
    """
    PyTorch Dataset for EMG data split into windows, where the first (window_size - 1) time steps
    are used as input and the last time step as the target.
    """
    def __init__(
        self, 
        inputs, 
        targets, 
        convert_dtype=False,
        enable_offset=False,
        offset=0
    ):
        """
         Args:
            inputs (np.ndarray): Input sequences.
            targets (np.ndarray): Target sequences.
            convert_dtype (bool): If true, convert the inputs and targets to long type (for classification).
            enable_offset (bool): If true, add an offset to the data.
            offset (int): The offset value.
        """
        if not isinstance(inputs, torch.Tensor):
            inputs = torch.tensor(inputs)
        if not isinstance(targets, torch.Tensor):
            targets = torch.tensor(targets)

        if convert_dtype:
            self.inputs = inputs.to(torch.long)
            self.targets = targets.to(torch.long)
        else:
            self.inputs = inputs
            self.targets = targets

        if enable_offset:
            assert isinstance(offset, int), f"Offset must be an integer, but got {type(offset)}"
            self.inputs = self.inputs + offset
            self.targets = self.targets + offset
    
    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]