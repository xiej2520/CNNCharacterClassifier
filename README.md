# CNN Character Classifier

## About

A Convolutional Neural Network (CNN) for classifying alphanumeric characters,
trained on the EMNIST database.

`[A-Z]|[a-z]|[0-9]`

[Static web page demo](https://xiej2520.github.io/CNNCharacterClassifier/)

1680s to train the model to 89.34% training accuracy, 87.87% test accuracy on a
GTX 1660 TI, 30 epochs.

## Prerequisites

Steps to run:

1. Install Python 3.10+
2. Install necessary libraries (numpy, opencv, torch (with cuda), torchvision)

   `pip install -r requirements.txt`

## Train/Test

To train a model, run

```Bash
python train.py
```

An output model will be exported at `./models/nnet.pt`.

Model architecture and hyperparameters can be tuned in `cnn.py` and `train.py`.
Uncomment some code in `train.py` to use a validation set during training.

To test the model at `./models/nnet.pt`, run

```Bash
python test.py
```

## To Run

`./models/nnet.pt` is required to run `main.py`, a pretrained one is provided.

`main.py` can be run with two options:

* ```Bash
  python main.py -f <file_name>
  ```

  runs the model on a single file.

* ```Bash
  python main.py -d <path> -o <outfile>
  ```

  runs the model on all the files in
`path` and outputs a result `outfile`. `-o` is optional, the output is written
in `path/pred.txt` by default.

Example image files have been provided in `examples`. Ideally, input image files
should be square, with black backgrounds and large white characters centered in
the image.

![Example C](examples/cCap.png) ![Example N](examples/nCap.png) ![Example N2](examples/nCap2.png)

## Run Web App

The model can be run in the browser with a locally hosted webpage.

1. A `./models/nnet.onnx` file is needed to run the model in the browser, and is
   provided. To generate a new one from `./models/nnet.pt`, run

   ```Bash
     python to_onnx.py
   ```

2. Run

   ```Bash
   python -m http.server
   ```

   to host the page, and navigate to `localhost:8000` in a web browser.
3. Provide input to the model by drawing on the canvas, or by uploading an image.
   * Drawn characters should preferably take up the entire canvas.
   * Uploaded images should preferably be white text on black background.

* The model currently appears to be producing inaccurate results on the browser.

![Web app screenshot](examples/webapp.png)
