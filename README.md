# AlexNet on CIFAR-10

A PyTorch implementation of a modified AlexNet architecture trained on the CIFAR-10 image classification dataset.


## Overview

AlexNet was originally introduced in the paper “ImageNet Classification with Deep Convolutional Neural Networks” and designed for large-scale image classification on ImageNet, which contains high-resolution images (typically 224×224) across 1000 classes. The original architecture demonstrated that deep convolutional networks could achieve breakthrough performance when trained on large datasets using GPUs and techniques like ReLU activation, dropout, and data augmentation.

In this implementation, AlexNet has been adapted for CIFAR-10, a smaller dataset consisting of 32×32 RGB images with 10 classes. Since CIFAR-10 images are significantly lower in resolution and complexity compared to ImageNet, the original architecture is modified by:

replacing large kernels (11×11, 5×5) with smaller 3×3 convolutions,
reducing fully connected layer size,
adjusting pooling operations to preserve spatial structure,
and tuning regularization for a smaller dataset.

These modifications preserve the core idea of AlexNet—hierarchical feature extraction using deep CNNs—while making it suitable for low-resolution datasets and faster training.
## Model Architecture

The model follows the standard AlexNet block structure with five convolutional layers and three fully connected layers, adjusted for CIFAR-10:

**Feature Extractor**

| Layer | Type | Out Channels | Kernel | Notes |
|-------|------|-------------|--------|-------|
| 1 | Conv2d + ReLU + BN + MaxPool | 64 | 3×3 | Spatial: 32 → 16 |
| 2 | Conv2d + ReLU + BN + MaxPool | 192 | 3×3 | Spatial: 16 → 8 |
| 3 | Conv2d + ReLU + BN | 384 | 3×3 | — |
| 4 | Conv2d + ReLU + BN | 256 | 3×3 | — |
| 5 | Conv2d + ReLU + BN + MaxPool | 256 | 3×3 | Spatial: 8 → 4 |

**Classifier**

```
Flatten → Linear(4096, 1024) → ReLU → Dropout(0.6)
       → Linear(1024, 512)   → ReLU → Dropout(0.6)
       → Linear(512, 10)
```

**Key design choices:**
- BatchNorm after every conv layer (replaces the original LRN)
- Dropout rate of 0.6 in the classifier
- Output: 10 logits for CIFAR-10 classes

## Dataset

[CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) — 60,000 32×32 colour images across 10 classes (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck), split into 50,000 training and 10,000 test images.

**Data augmentation (training only):**
- Random horizontal flip
- Random crop (32×32 with padding=4)

## Requirements

```
torch
torchvision
torchinfo
```


The CIFAR-10 dataset will be automatically downloaded to a `data/` folder on first run.

**Training configuration:**

| Parameter | Value |
|-----------|-------|
| Optimizer | Adam |
| Learning rate | 0.001 |
| Weight decay | 0.0001 |
| Batch size | 1024 |
| Epochs | 19 |
| Loss function | CrossEntropyLoss |

Training runs on GPU automatically if CUDA is available, otherwise falls back to CPU.

## Output

After training, the script prints per-epoch loss and final test accuracy:

```
epoch0 loss: 1.4832
epoch1 loss: 1.1204
...
epoch18 loss: 0.4571
accuracy: 84.37
```
##Size 

Total parameters: 6,979,146 

Estimated total size: 30.69 MB

## Notes

- The model uses `train=False, download=False` for the test set — ensure the training set has been downloaded first before running evaluation standalone.
- Shuffle is enabled on the test loader; set it to `False` for deterministic evaluation.
- To save the trained model, add `torch.save(model.state_dict(), "alexnet_cifar10.pth")` after training.
