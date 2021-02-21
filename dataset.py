"""Data utility functions."""
import os
import numpy as np
import nibabel as nib
import torch
import torch.utils.data as data
from PIL import Image
from torchvision import transforms


class HeartData(data.Dataset):

    def __init__(self, image_paths_file):
        self.image_paths_file = image_paths_file
        self.only_inputs = image_paths_file["input"]

    def __getitem__(self, idx):
        # define transformations
        to_normalise = transforms.Normalize([0.1278], [0.1814])
        to_tensor = transforms.ToTensor()

        # open input image
        image_name = self.image_paths_file["path_files"] + self.image_paths_file["input"][idx]  # join folder and file
        image = Image.open(image_name)
        image = np.array(image)
        image = to_tensor(image)
        # image = to_normalise(image)

        # open target volume
        target_name = self.image_paths_file["target"][self.image_paths_file["input"][idx][0:4]]
        buffer = nib.load(target_name)
        target = np.array(buffer.dataobj)
        target = target.astype('float32')
        target = to_tensor(target)

        return image, target

    def __len__(self):
        return len(self.only_inputs)
