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
        self.age_mean = 25.86
        self.age_std = 8.61

        self.afad_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                [0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        item = self.data_list[idx]
        image_path = item[0]
        # print(image_path)
        gender_label = item[1]
        age_label = item[2]
        catid = item[3]

        age_label = (age_label - self.age_mean) / self.age_std

        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Image not found at {image_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (224, 224))

        # 转换为 one-hot 向量
        # if gender_label == 1:  # 男性
        #     gender_label_onehot = [1.0, 0.0]
        # else:  # 女性
        #     gender_label_onehot = [0.0, 1.0]

        # Apply transformations
        is_afad = 'afad' in image_path.lower()
        if is_afad:
            image = self.afad_transform(image)
        elif self.transform:
            image = self.transform(image)

        return (
            image,
            torch.tensor(gender_label, dtype=torch.long),
            torch.tensor(age_label, dtype=torch.float32)
        )

class AddGaussianNoise(object):
    def __init__(self, mean=0., std=0.01, p=0.5):
        self.mean = mean
        self.std = std
        self.p = p

    def __call__(self, tensor):
        if random.random() < self.p:
            noise = torch.randn(tensor.size()) * self.std + self.mean
            return tensor + noise
        return tensor

    def __repr__(self):
        return f"{self.__class__.__name__}(mean={self.mean}, std={self.std}, p={self.p})"

# Data Augmentation and Preprocessing
data_transforms = {
    'train': transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomResizedCrop(224, scale=(0.6, 1.0)),  # 裁剪範圍更大
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),  # 旋轉角度加大
        transforms.RandomPerspective(distortion_scale=0.3, p=0.5),  # 隨機透視變形
        transforms.ColorJitter(
            brightness=(0.5, 1.5),  # 更廣亮度變化
            contrast=(0.5, 1.5),
            saturation=(0.5, 1.5),
            hue=0.2
        ),
        transforms.RandomGrayscale(p=0.2),  # 灰階機率略升
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.5)),  # sigma範圍加大
        transforms.ToTensor(),
        AddGaussianNoise(mean=0., std=0.03, p=0.7),  # 噪聲稍強
        transforms.Normalize([0.485, 0.456, 0.406],
                            [0.229, 0.224, 0.225]),
    ]),
    'val': transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])
}

def get_dataloader():
    
    train_data, val_data = get_training_data()
    
    # Create Datasets
    train_dataset = AgeGenderDataset(train_data, transform=data_transforms['train'])
    val_dataset = AgeGenderDataset(val_data, transform=data_transforms['val'])

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=96, shuffle=True, num_workers=6, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=96, shuffle=False, num_workers=6, pin_memory=True)

    # # Example usage
    # for images, gender_labels, age_labels in train_loader:
    #     print(images.shape, gender_labels.shape, age_labels.shape)
    #     break

    return train_loader, val_loader, train_data

