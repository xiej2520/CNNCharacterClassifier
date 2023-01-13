import torch
import torch.nn as nn
import torch.utils.data.dataloader as DL
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

import time

from torchvision.datasets import EMNIST

from cnn import CNN

def train_net(nnet, train_dl, valid_dl=None):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    nnet = nnet.to(device)

    nnet.train()

    epochs = 30
    learn_rate = 0.0005

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(nnet.parameters(), lr=learn_rate)

    def validate():
        nnet.eval()
        with torch.no_grad():
            run_loss, run_correct, run_total = 0, 0, 0
            for images, labels in valid_dl:
                images = images.to(device)
                labels = labels.to(device)

                outputs = nnet(images.float())
                loss = criterion(outputs, labels)

                _, pred = torch.max(outputs.data, 1) # not labels
                run_total += labels.size(0)
                run_correct += (pred == labels).sum().item()

                run_loss += loss.item()
            accuracy = run_correct / run_total
            print("Validation Loss: {:.2f}, Accuracy: {:.2f}%".format(run_loss, accuracy * 100))

    start = time.time()

    for i in range(epochs):
        nnet.train()

        run_loss, run_correct, run_total = 0, 0, 0
        for images, labels in train_dl:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()

            outputs = nnet(images.float())
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            _, pred = torch.max(outputs.data, 1)
            run_total += labels.size(0)
            run_correct += (pred == labels).sum().item()

            run_loss += loss.item()
        accuracy = run_correct / run_total
        print("Epoch: {}, Loss: {:.2f}, Accuracy: {:.2f}%".format(i, run_loss, accuracy * 100))
        if valid_dl != None:
            validate()

    print("Time taken to train: {:.2f}s".format(time.time() - start))
    torch.save(nnet.state_dict(), "./models/nnet.pt")


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

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train_data.data.to(device)

    """
    v = 50000 # size of validation dataset
    train_data, valid_data = random_split(train_data, [len(train_data) - v, v])
    """

    #print("Training: {}, Validation: {}, Test: {}".format(len(train_data), v, len(test_data)))
    print("Training dataset size: {}".format(len(train_data)))

    train_dl = DL.DataLoader(train_data, batch_size=500, shuffle=True, num_workers=8, pin_memory=True)
    #valid_dl = DL.DataLoader(valid_data, batch_size=500, shuffle=True, num_workers=8, pin_memory=True)

    nnet = CNN()
    train_net(nnet, train_dl)
