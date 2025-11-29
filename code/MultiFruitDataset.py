from __future__ import print_function
import os
import torch
from torch.utils.data import Dataset
from PIL import Image

class MultiFruitDataset(Dataset):
    def __init__(self, image_folder, label_file, transform=None):
        self.image_folder = image_folder
        self.transform = transform

        # Load labels
        self.image_paths = []
        self.labels = []

        # Open the label txt file
        with open(label_file, 'r') as f:
            for line in f:
                # Get image file name and label string
                parts = line.strip().split()
                img_name = parts[0]
                label_str = parts[1]

                # Convert label string to float list
                label_vector = [float(c) for c in label_str]

                # Create path to image and append to list
                self.image_paths.append(os.path.join(image_folder, img_name))
                
                # Convert label list to PyTorch tensor and append it
                self.labels.append(torch.tensor(label_vector, dtype=torch.float))
    
    def __len__(self):
        # Returns the total number of images in dataset
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert('RGB')

        if self.transform:
            img = self.transform(img)
        
        label = self.labels[idx]

        return img, label