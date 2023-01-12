import torch
import torch.utils.data.dataloader as DL
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

from torchvision.datasets import EMNIST

from cnn import CNN

def test_net(nnet, test_dl):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    nnet = nnet.to(device)

    nnet.eval()
    with torch.no_grad():
        run_total, run_correct = 0, 0
        for images, labels in test_dl:
            images = images.to(device)
            labels = labels.to(device)
            outputs = nnet(images.float())
            _, pred = torch.max(outputs.data, 1) # not labels

            run_total += labels.size(0)
            run_correct += (pred == labels).sum().item()
        print("Test batch accuracy: {}/{} = {:.2f}%"
            .format(run_correct, run_total, run_correct / run_total * 100)
        )


def tfrot(img):
    return TF.rotate(img, 90)
def tfvflip(img):
    return TF.vflip(img)

if __name__ == "__main__":
    nnet = CNN()
    nnet.load_state_dict(torch.load("nnet.pt"))

    test_data = EMNIST(root="./data", split="byclass", train=False, download=True,
        transform=transforms.Compose([
            tfrot,
            tfvflip,
            transforms.ToTensor()
        ]))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    test_data.data.to(device)

    print("Test dataset size {}".format(len(test_data)))
    test_dl = DL.DataLoader(test_data, batch_size=500, shuffle=True, num_workers=8, pin_memory=True)

    test_net(nnet, test_dl)
