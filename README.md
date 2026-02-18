# CNN for Handwritten Digit Classification (MNIST)

## Overview
This project implements a Convolutional Neural Network (CNN) in MATLAB to classify handwritten digits from the MNIST dataset.

The model achieves approximately **98% test accuracy** on the MNIST dataset.

---

## Architecture
The CNN architecture includes:

- Convolution Layer  
- ReLU Activation  
- Pooling Layer  
- Fully Connected Layer  
- Softmax Output Layer  

---

## Project Structure

- `LoadMNISTimages.m` – Loads image data from the dataset  
- `LoadMNISTlabels.m` – Loads label data from the dataset  
- `Conv.m` – Convolution operation  
- `ReLU.m` – ReLU activation function  
- `Pool.m` – Pooling layer implementation  
- `Softmax.m` – Output classification layer  
- `MnistConv.m` – Main training script  
- `PlotFeatures.m` – Visualizes extracted feature maps  
- `TestMnist.m` – Evaluates model accuracy  
- `DATASET/` – MNIST dataset files  

---

## Results

- Final Test Accuracy: **~98%**
- Feature maps and intermediate activations can be visualized.
- The network successfully learns hierarchical features from handwritten digit images.

---

### ▶ To Visualize Feature Extraction Process

Run: PlotFeatures
This displays intermediate feature maps and visualizes how the CNN processes the input image.

---

### ▶ To Test Model Accuracy

Run:  TestMnist
This evaluates the trained model and prints the final classification accuracy.

---

## Author

Namit Sahu  
B.Tech – Electronics and Communication Engineering
230102059


