# Digit Recognition using Faster R-CNN

## Student Information
- **Student ID**: 110550020
- **Name**: Enfu Liao (廖恩莆)

## Description

This repo contains the implementation for digit recognition using the Faster R-CNN framework.

## Setup and Usage

```bash
pip install -r requirements.txt --index-url https://download.pytorch.org/whl/cu126 --extra-index-url https://pypi.org/simple --break-system-packages
```

```bash
# training
python -m nycu_cv_hw2.train
```

```bash
# inference
python -m nycu_cv_hw2.test
```

## Repository Structure

```
├── data/              # Dataset used for training/testing (if applicable)
│   ├── train          # Training data (images)
│   ├── train.json     # Training data (labels in COCO format)
│   ├── valid          # Validation data (images)
│   ├── valid.json     # Validation data (labels in COCO format)
│   └── test           # Test data (images)
├── nycu_cv_hw2/       # Main package containing source code
├── settings.toml      # Configuration file for training (e.g., hyperparameters)
├── models/            # Model
├── outputs/           # TODO
├── logs/              # Training history
├── basic.log          # TODO
├── README.md          # This file
└── requirements.txt   # Required dependencies
```

## Results

<!-- TODO -->
