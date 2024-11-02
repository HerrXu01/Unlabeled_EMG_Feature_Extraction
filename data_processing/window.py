import random
import numpy as np

class SlidingWindowSplitter:
    """
    A class to apply sliding window to time-series data.
    """

    def __init__(self, config: dict, data: np.ndarray):
        self.config = config
        self.data = data

    def fixed_step_window(self):
        """
        Applies sliding window to the given data.
        
        Returns:
            np.ndarray: The windowed data with shape (num_windows, window_size, num_channels).
        """
        if not isinstance(self.data, np.ndarray):
            raise ValueError("Input data should be a numpy array.")

        windows = []
        window_size = self.config["window"]["window_size"]
        step_size = self.config["window"]["step_size"]
        
        for start in range(0, len(self.data) - window_size + 1, step_size):
            end = start + window_size
            window = self.data[start:end, :]
            windows.append(window)
        
        return np.array(windows)

    def random_step_window(self):
        """
        Applies sliding window to the given data with a random step size between specified bounds.
        
        Returns:
            np.ndarray: The windowed data with shape (num_windows, window_size, num_channels).
        """
        if not isinstance(self.data, np.ndarray):
            raise ValueError("Input data should be a numpy array.")

        windows = []
        window_size = self.config["window"]["window_size"]
        min_step_size = self.config["window"]["min_step_size"]
        max_step_size = self.config["window"]["max_step_size"]
        
        start = 0
        while start <= len(self.data) - window_size:
            end = start + window_size
            window = self.data[start:end, :]
            windows.append(window)
            step_size = random.randint(min_step_size, max_step_size)
            start += step_size
        
        return np.array(windows)
        
