# U-Net Segmentation with PyTorch

This repository provides an implementation of a U-Net model in PyTorch for semantic segmentation tasks. The implementation includes custom loss functions, data preprocessing pipelines, and a training loop for performing image segmentation.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Requirements](#requirements)
- [Data Structure](#data-structure)
- [Usage](#usage)
  - [Training](#training)
  - [Visualization](#visualization)
- [Model Architecture](#model-architecture)
- [Custom Loss Functions](#custom-loss-functions)
- [Acknowledgments](#acknowledgments)

---

## Introduction

The U-Net is a convolutional neural network designed for biomedical image segmentation. This implementation uses PyTorch and supports data augmentation with the Albumentations library.

## Features

- U-Net architecture for image segmentation
- Dice+BCE loss function and IoU loss
- Data augmentation with Albumentations
- GPU support for training
- Real-time visualization of training progress

## Requirements

Install the required dependencies:

```bash
pip install numpy opencv-python torch torchvision matplotlib albumentations
```

## Data Structure

The project assumes the following directory structure for training and testing data:

```
stage1_train/
  |-- <folder1>/
      |-- images/
          |-- <image_file>
      |-- masks/
          |-- <mask_file1>
          |-- <mask_file2>
  |-- <folder2>/
      |-- ...
stage1_test/
  |-- <folder1>/
      |-- images/
          |-- <image_file>
  |-- <folder2>/
      |-- ...
```

- **images/** contains the input images.
- **masks/** contains corresponding segmentation masks.

## Usage

### Training

To train the U-Net model, simply run the `Main` class:

```bash
python <script_name>.py
```

### Visualization

The script will display real-time plots showing:

- Training loss over epochs
- Comparison of the original image, predicted mask, and ground truth mask

![](https://github.com/jason19990305/U_Nnet/blob/main/Image/upload_45ca665f9e336ec8f6c15925b5783eba.png)

## Model Architecture

The U-Net implementation includes the following components:

- Encoder: Downsampling layers using convolution and max-pooling.
- Bottleneck: A single convolutional layer connecting encoder and decoder.
- Decoder: Upsampling layers with transposed convolution and concatenation.
- Output: A final convolution layer that maps features to the segmentation mask.

### Example Architecture Diagram

```
Input (128x128x3) -> Encoder -> Bottleneck -> Decoder -> Output (128x128x1)
```

## Custom Loss Functions

This implementation includes:

1. **Dice+BCE Loss**: Combines Binary Cross Entropy (BCE) with Dice loss for improved segmentation accuracy.
2. **IoU Loss**: Calculates Intersection over Union (IoU) to evaluate model performance.


