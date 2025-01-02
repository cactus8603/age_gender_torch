import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import timm
import cv2
import numpy as np
import csv

from utils import get_training_data

# Custom Dataset
class AgeGenderDataset(Dataset):
    def __init__(self, data_list, transform=None):
        """
        Args:
            data_list (list): List of data items where each item contains
                              [image_path, gender_label, age_label].
            transform (callable, optional): Transform to be applied on a sample.
        """
        self.data_list = data_list
        self.transform = transform

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        item = self.data_list[idx]
        image_path = item[0]
        # print(image_path)
        gender_label = item[1]
        age_label = item[2]

        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Image not found at {image_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (224, 224))

        # 转换为 one-hot 向量
        if gender_label == 1:  # 男性
            gender_label_onehot = [1.0, 0.0]
        else:  # 女性
            gender_label_onehot = [0.0, 1.0]

        # Apply transformations
        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(gender_label, dtype=torch.long), torch.tensor(age_label, dtype=torch.float32)

# Helper function to load dataset from multiple directories
# def load_data_from_directories(directories, valid_extensions={'.jpg', '.png'}, gender_mapping=None):
#     """
#     Args:
#         directories (list): List of root directories containing image data.
#         valid_extensions (set): Set of valid image file extensions.
#         gender_mapping (callable): Function to determine gender label from path or metadata.
#     Returns:
#         data_list (list): List of [image_path, gender_label, age_label].
#     """
#     data_list = []
#     for directory in directories:
#         for root, _, files in os.walk(directory):
#             for file in files:
#                 if os.path.splitext(file)[1].lower() in valid_extensions:
#                     image_path = os.path.join(root, file)
#                     # Extract gender and age from filename or metadata (example placeholder logic)
#                     gender_label = gender_mapping(image_path) if gender_mapping else 0  # Default gender_label
#                     age_label = random.uniform(0, 1)  # Placeholder for normalized age
#                     data_list.append([image_path, gender_label, age_label])
#     return data_list

# Helper function to load data from CSV
# def load_data_from_csv(csv_path):
#     """
#     Args:
#         csv_path (str): Path to the CSV file.
#     Returns:
#         data_list (list): List of [image_path, gender_label, age_label].
#     """
#     data_list = []
#     with open(csv_path, mode='r', encoding='utf-8') as file:
#         csv_reader = csv.reader(file)
#         next(csv_reader)  # Skip the header
#         for row in csv_reader:
#             if '-1' not in row and float(row[1]) < 90:  # Filtering condition
#                 image_path = row[0]
#                 gender_label = int(row[2])  # 0 or 1
#                 age_label = float(row[1]) / 100.0  # Normalize age
#                 data_list.append([image_path, gender_label, age_label])
#     return data_list

# Example gender mapping function
# def example_gender_mapping(image_path):
#     """Example function to determine gender based on file naming or folder structure."""
#     if 'male' in image_path.lower():
#         return 1
#     elif 'female' in image_path.lower():
#         return 0
#     return 0

# Data Augmentation and Preprocessing
data_transforms = {
    'train': transforms.Compose([
        transforms.ToPILImage(),  # 将 numpy.ndarray 转为 PIL.Image
        transforms.Resize(256),
        # transforms.RandomHorizontalFlip(),
        # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
    'val': transforms.Compose([
        transforms.ToPILImage(),  # 将 numpy.ndarray 转为 PIL.Image
        transforms.Resize(256),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
}

def get_dataloader():
    
    train_data, val_data = get_training_data()
    
    # Create Datasets
    train_dataset = AgeGenderDataset(train_data, transform=data_transforms['train'])
    val_dataset = AgeGenderDataset(val_data, transform=data_transforms['val'])

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=4)

    # # Example usage
    # for images, gender_labels, age_labels in train_loader:
    #     print(images.shape, gender_labels.shape, age_labels.shape)
    #     break

    return train_loader, val_loader

