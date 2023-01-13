import cv2
import torch
import torchvision.transforms as transforms

from cnn import CNN

def load_image(path):
    img = cv2.resize(cv2.imread(path, cv2.IMREAD_GRAYSCALE), (28, 28))
    img = transforms.ToTensor()(img)
    img = img.unsqueeze(0)
    img = transforms.Normalize(mean=0.1736, std=0.3316)(img)
    return img

if __name__ == "__main__":
    nnet = CNN()
    nnet.load_state_dict(torch.load("./models/nnet.pt"))
    nnet.eval()

    img = load_image("./examples/0.png")

    torch.onnx.export(nnet,
            img,
            "./models/nnet.onnx",
            export_params=True,
            input_names=["in"],
            output_names=["out"],
            verbose=True
            )
