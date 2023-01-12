# CNN Character Classifier

A Convolutional Neural Network (CNN) for classifying alphanumeric characters,
trained on the EMNIST database.

`[A-Z]|[a-z]|[0-9]`

1680s to train the model to 89.34% training accuracy, 87.87% test accuracy on a
GTX 1660 TI, 30 epochs.

Steps to run:

1. Install Python 3.10+
2. Install necessary libraries (numpy, opencv, torch (with cuda), torchvision)

   `pip install -r requirements.txt`

To train a model, run `python train.py`. An output model will be exported at
`nnet.pt`. Model architecture and hyperparameters can be tuned in `cnn.py` and
`train.py`. Uncomment some code in `train.py` to use a validation set during
training.

To test the model at `nnet.pt`, run `python test.py`.

`main.py` can be run with two options:

* `python main.py -f <file_name>` runs the model on a single file.
* `python main.py -d <path> -o <outfile>` runs the model on all the files in
`path` and outputs a result `outfile`. `-o` is optional, the output is written
in `path/pred.txt` by default.

Example image files have been provided in `examples`. Ideally, input image files
should be square, with black backgrounds and large white characters centered in
the image.
