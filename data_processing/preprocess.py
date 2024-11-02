import os
import pandas as pd
import numpy as np
from scipy.io import loadmat
from common.registry import registry
import data_processing.filter
from data_processing.window import SlidingWindowSplitter
import data_processing.load_raw_data

class EMGPreprocessor:
    
    def __init__(self, config: dict):
        self.config = config
        self.data_frames = []

    def load_data(self):
        """
        Loads EMG data specified in the configuration.
        """
        self.data_frames = registry.get_rawdataloader(
            self.config["dataset"]["name"]
        )(self.config)

    def filter_data(self):
        """
        Applies a filter to the EMG data based on the choice in the configuration.
        """
        if not self.data_frames:
            raise ValueError("No data loaded. Please load the data first.")

        filtered_data_frames = []
        for data in self.data_frames:
            filter_instance = registry.get_filter_class(
                self.config["filter"]["type"]
            )(
                config=self.config,
                data=data,
            )
            filtered_data = filter_instance.apply()
            filtered_data_frames.append(filtered_data)
        
        self.data_frames = filtered_data_frames

    def sliding_window(self):
        """
        Applies a sliding window to the EMG data.
        """
        if not self.data_frames:
            raise ValueError("No data loaded. Please load the data first.")
        
        all_windows = []
        window_type = self.config["window"].get("type", "fixed_step")
        for data in self.data_frames:
            data_window = SlidingWindowSplitter(config=self.config, data=np.array(data))
            if window_type == "fixed_step":
                windows = data_window.fixed_step_window()
            elif window_type == "random_step":
                windows = data_window.random_step_window()
            else:
                raise ValueError("Invalid window type specified in config. Use 'fixed_step' or 'random_step'.")
            
            all_windows.append(windows)
        
        return np.concatenate(all_windows, axis=0)

    def save_windows(self, windows):
        """
        Save the windows as a .npy file
        """
        windows_dir = self.config["window"]["windows_dir"]
        filename = self.config["window"]["filename"]
        os.makedirs(windows_dir, exist_ok=True)
        save_path = os.path.join(windows_dir, filename)
        np.save(save_path, windows)

        print(f"Sliding windows data saved successfully at: {save_path}")

    def process(self):
        """
        Combine the previous process.
        """
        self.load_data()
        
        if self.config["data_preprocess"]["enable_filter"]:
            self.filter_data()

        if self.config["data_preprocess"]["enable_window"]:
            windows = self.sliding_window()
        else:
            # Return the full sequecnce data.
            return self.data_frames

        if self.config["data_preprocess"]["enable_save_windows"]:
            self.save_windows(windows)

        print(f"The full windows data is of the shape {windows.shape}.")

        return windows