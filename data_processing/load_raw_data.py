import os
import pandas as pd
from scipy.io import loadmat
from common.registry import registry

@registry.register_rawdataloader("Ninapro")
def ninapro_loader(config: dict):
    """
    Loads Ninapro EMG data from the specified directory.
    """
    raw_data_path = config["dataset"]["src"]
    data_frames = []

    for file_name in os.listdir(raw_data_path):
        file_path = os.path.join(raw_data_path, file_name)

        try:
            try:
                mat_data = loadmat(file_path)
                emg_data = mat_data[config["dataset"]["data_key"]]
            except Exception as e:
                raise ValueError(f"Error loading .mat file: {e}")
            data_frame = pd.DataFrame(emg_data)
            data_frames.append(data_frame)
        except ValueError as e:
            print(f"Skipping file {file_name}: {e}")

        if not data_frames:
            raise ValueError("No valid data files were found in the directory.")

    return data_frames

@registry.register_rawdataloader("SeNic")
def senic_loader(config: dict):
    """
    Loads SeNic EMG data from the specified directory.
    """
    raw_data_path = config["dataset"]["src"]
    data_frames = []

    for sub_dir_name in os.listdir(raw_data_path):
        sub_dir_path = os.path.join(raw_data_path, sub_dir_name)

        if os.path.isdir(sub_dir_path):
            for file_name in os.listdir(sub_dir_path):
                file_path = os.path.join(sub_dir_path, file_name)

                try:
                    if file_name.endswith('.csv'):
                        data_frame = pd.read_csv(file_path, header=None)
                        data_frames.append(data_frame)
                except Exception as e:
                    print(f"Skipping file {file_name} in {sub_dir_name}: {e}")

    if not data_frames:
        raise ValueError("No valid data files were found in the subdirectories.")

    return data_frames