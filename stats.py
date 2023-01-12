import numpy as np

import torch
import torch.nn as nn
import torch.utils.data.dataloader as DL
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

from torchvision.datasets import EMNIST

def tfrot(img):
    return TF.rotate(img, 90)
def tfvflip(img):
    return TF.vflip(img)

if __name__ == "__main__":
    train_data = EMNIST(root="./data", split="byclass", train=True, download=True,
    transform=transforms.Compose([
        tfrot,
        tfvflip,
        transforms.ToTensor()
    ]))

    train_dl = DL.DataLoader(train_data, batch_size=100, shuffle=False, num_workers=8, pin_memory=True)

    sum_means = 0
    sum_std = 0
    batches = 0
    for data, _ in train_dl:
        batches += 1
        s, ss = 0, 0
        for img in data:
            s += img.sum()
            ss += (img ** 2).sum()
        mean = s / (28 * 28 * 100)
        sum_means += mean
        std = torch.sqrt((ss / (28 * 28 * 100)) - mean ** 2)
        sum_std += std

    print(batches)
    print("Mean of means: " + str(sum_means / batches)) # 0.1736
    print("Stdev: " + str(sum_std / batches)) # 0.3316
