# SPDX-FileCopyrightText: 2024 Helmholtz Centre Potsdam - GFZ German Research Centre for Geosciences
# SPDX-FileCopyrightText: 2024 Ekin Celikkan <ekin.celikkan@gfz-potsdam.de>
# SPDX-License-Identifier: Apache-2.0

from PIL import Image
import torch
from torch.utils.data import Dataset
import os
import numpy as np
import matplotlib.pyplot as plt
from utils import augment_data


class WeedsGaloreDataset(Dataset):
    def __init__(self, dataset_path, dataset_size, in_bands, num_classes, is_training, split, augmentation):
        self.img_dir = os.path.join(dataset_path)
        self.in_bands = in_bands
        self.num_classes = num_classes
        self.is_training = is_training
        self.augmentation = augmentation
        self.split = split


        with open(os.path.dirname(os.path.abspath(__file__)) + f'/splits/{self.split}.txt', 'r') as file:
            data = [line.rstrip('\n') for line in file]  # Assuming elements are numeric
        self.img_list = np.array(data)

        if self.is_training:
            self.dataset_size = dataset_size
        else:
            self.dataset_size = self.img_list.size  # always take full validation list

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # load single bands and construct input image
        img_path = os.path.join(self.img_dir, self.img_list[idx][:10], 'images', self.img_list[idx])
        red_band = plt.imread(img_path + '_R.png')
        green_band = plt.imread(img_path + '_G.png')
        blue_band = plt.imread(img_path + '_B.png')
        nir_band = plt.imread(img_path + '_NIR.png')
        re_band = plt.imread(img_path + '_RE.png')
        if self.in_bands == 3:
            img = np.stack((red_band, green_band, blue_band))
        elif self.in_bands == 5:
            img = np.stack((red_band, green_band, blue_band, nir_band, re_band))

        # load semantic label
        label_path = os.path.join(self.img_dir, self.img_list[idx][:10], 'semantics', self.img_list[idx])
        label = Image.open(label_path + '.png')
        label = np.array(label)

        # binary label (bg:0, crop:1, weed:2)
        binary_label = np.copy(label)
        binary_label[binary_label > 1] = 2

        # Data Augmentation
        if self.is_training and self.augmentation:
            img, label, binary_label = augment_data(img, label, binary_label)

        return img, label, binary_label
