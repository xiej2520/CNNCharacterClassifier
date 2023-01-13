import cv2
import numpy as np
import matplotlib.pyplot as plt

import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

from cnn import CNN

import sys
import os

char = {}
idx = {}
for i in range(10):
    char[i] = str(i)
    idx[str(i)] = i
for i in range(10, 36):
    char[i] = chr(i + 55)
    idx[chr(i + 55)] = i
for i in range(36, 62):
    char[i] = chr(i + 61)
    idx[chr(i + 61)] = i

def tfrot(img):
    return TF.rotate(img, 90)
def tfvflip(img):
    return TF.vflip(img)

def pred(nnet, img):
    nnet.eval()
    with torch.no_grad():
        output = nnet(img.float())
        val, index = torch.max(output.data, 1)
        return index.item()

def load_image(path):
    img = cv2.resize(cv2.imread(path, cv2.IMREAD_GRAYSCALE), (28, 28))
    img = transforms.ToTensor()(img)
    img = img.unsqueeze(0)
    img = transforms.Normalize(mean=0.1736, std=0.3316)(img)
    return img

if __name__ == "__main__":

    nnet = CNN()
    nnet.load_state_dict(torch.load("./models/nnet.pt"))


    if len(sys.argv) == 3 and sys.argv[1] == "-f":
        img = load_image(sys.argv[2])
        print("Prediction is:", char[pred(nnet, img)])
        plt.imshow(np.squeeze(img), cmap="gray")
        plt.show()
    elif len(sys.argv) >= 3 and sys.argv[1] == "-d":
        path = os.path.abspath(sys.argv[2])
        files = os.listdir(sys.argv[2])
        outfile = sys.argv[2] + "/pred.txt"
        if len(sys.argv) == 5 and sys.argv[3] == "-o":
            outfile = sys.argv[4]
        with open(outfile, 'w') as out:
            for imgfile in files:
                try:
                    img = load_image("{}/{}".format(path, imgfile))
                    out.write("{}: {}\n".format(char[pred(nnet, img)], imgfile))
                except:
                    out.write("Failed: {}\n".format(imgfile))

    """
    test_data = EMNIST(root="./data", split="byclass", train=False, download=True,
        transform=transforms.Compose([
            tfrot,
            tfvflip,
            transforms.ToTensor()
        ]))
    labels = sorted(test_data.classes)
    print("Classes: {}".format(labels))
    print("Number of classes: {}".format(len(labels)))
    """

    """
    def show_image(data):
        img, label = data
        print("Label: (" + str(char[label]) + ")")
        plt.imshow(img[0], cmap="gray")
        plt.show()

    show_image(test_data[100])

    test_dl = DL.DataLoader(test_data, batch_size=500, shuffle=True, num_workers=8, pin_memory=True)

    def show_batch(dl):
        for images, labels in dl:
            fig, ax = plt.subplots(figsize=(12, 12))
            ax.set_xticks([]); ax.set_yticks([])
            ax.imshow(make_grid(images, nrow=20).permute(1, 2, 0))
            plt.show()
            break

    show_batch(test_dl)
    """
