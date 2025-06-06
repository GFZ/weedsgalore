# SPDX-FileCopyrightText: 2024 Helmholtz Centre Potsdam - GFZ German Research Centre for Geosciences
# SPDX-FileCopyrightText: 2024 Ekin Celikkan <ekin.celikkan@gfz-potsdam.de>
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import cv2


def augment_data(img, label, binary_label):
    """image: of shape (h,w,c)"""
    # 1. Rotate around upright axis-axis (z-axis)
    angle = np.random.choice([0, 1, 2, 3])
    img = np.rot90(img, k=angle, axes=(1, 2))
    label = np.rot90(label, k=angle)
    binary_label = np.rot90(binary_label, k=angle)
    # 2. Flip
    flip = np.random.choice([0, 1, 2])  # 0 - no flip, 1 - horizontal flip, 2 - vertical flip
    if flip == 1:
        img = img[:, ::-1, :]
        label = cv2.flip(label, flip-1)
        binary_label = cv2.flip(binary_label, flip-1)
    elif flip == 2:
        img = img[:, :, ::-1]
        label = cv2.flip(label, flip)
        binary_label = cv2.flip(binary_label, flip)
    # 3. Add random noise
    jitter = np.random.randn(img.shape[0], img.shape[1], img.shape[2]) / 25   # sample from Gaussian and divide by a factor
    img = img + jitter
    img = img.astype(np.float32)

    img_copy, label_copy, binary_label_copy = img.copy(), label.copy(), binary_label.copy()   # copy() added to avoid negative stride

    return img_copy, label_copy, binary_label_copy


