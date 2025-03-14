import io
import os
from glob import glob

import requests
import zipfile
import tarfile
import pickle

import scipy
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset


class SpeechComand(TensorDataset):
    def __init__(self, path: str = "data"):
        # Download the Speech Command dataset
        if not os.path.exists(os.path.join(path, "sc09")):
            url = "https://huggingface.co/datasets/krandiash/sc09/resolve/main/sc09.zip"
            print("Downloading dataset...")
            response = requests.get(url)
            if response.status_code == 200:
                with zipfile.ZipFile(io.BytesIO(response.content)) as zip_ref:
                    zip_ref.extractall(path)
                print("Dataset downloaded and extracted successfully.")
            else:
                print("Failed to download the Speech Command dataset.")

        # Load the dataset to memory
        data = []
        labels = []
        for i, number in enumerate(
            ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
        ):
            for file_path in glob(os.path.join(path, "sc09", number, "*.wav")):
                sample_rate, waveform = scipy.io.wavfile.read(file_path)
                assert sample_rate == 16000, "Sample rate is expected to be 16kHz"
                assert len(waveform) <= 16000, "Audio data is expected to be less than 1 second"
                assert waveform.dtype == np.int16, "Audio data expected to be in int16 format"
                waveform = np.pad(waveform, (0, 16 * 1024 - waveform.shape[-1]))
                data.append(waveform)
                labels.append(i)

        # Standardize and convert to tensors
        data = np.stack(data)[..., None] / 32768.0
        data = data / data.std(-2, keepdims=True)
        labels = np.stack(labels)
        data = torch.as_tensor(data, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        super().__init__(data, labels)

    def train_dataloader(self, batch_size: int):
        return DataLoader(self, batch_size, shuffle=True, num_workers=1, drop_last=True)


class CIFAR10Dataset(TensorDataset):
    def __init__(self, path: str = "data"):
        # Download the CIFAR-10 dataset
        if not os.path.exists(os.path.join(path, "cifar10")):
            url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
            print("Downloading CIFAR-10 dataset...")
            response = requests.get(url)
            if response.status_code == 200:
                with tarfile.open(fileobj=io.BytesIO(response.content), mode="r:gz") as tar_ref:
                    tar_ref.extractall(path)
                print("CIFAR-10 dataset downloaded and extracted successfully.")
            else:
                print("Failed to download the CIFAR-10 dataset.")

        # Load the dataset to memory
        data = []
        labels = []
        for idx in [1, 2, 3, 4, 5]:
            with open(os.path.join("data", "cifar-10-batches-py", f"data_batch_{idx}"), "rb") as fo:
                batch = pickle.load(fo, encoding="bytes")
            assert batch[b"data"].dtype == np.uint8, "Unexpected data type"
            assert batch[b"data"].shape == (10000, 3072), "Unexpected shape of data"
            assert len(batch[b"labels"]) == 10000, "Unexpected length of labels"
            data.append(batch[b"data"].reshape(-1, 3, 32, 32))
            labels.append(np.array(batch[b"labels"]))

        # convert to tensors
        data = np.concatenate(data) / 255.0
        labels = np.concatenate(labels)
        data = torch.as_tensor(data, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        super().__init__(data, labels)

    def train_dataloader(self, batch_size: int):
        return DataLoader(self, batch_size, shuffle=True, num_workers=1, drop_last=True)
