# Age and Gender Prediction using PyTorch

This repository provides a PyTorch-based implementation for training, evaluating, and deploying an age and gender prediction model. It includes modules for dataset preparation, model architecture, training, and utility functions to simplify the workflow.

---

## Features

- **Flexible Dataset Handling**: Supports multiple datasets with customizable preprocessing pipelines.
- **Custom Model**: Implements a multi-task learning model for gender classification and age regression.
- **Training Pipeline**: Includes mixed precision training, checkpoint saving, and TensorBoard integration.
- **CUDA Support**: Optimized for GPU acceleration to speed up training and inference.

---

## File Descriptions

### 1. `dataset.py`

This file defines the `AgeGenderDataset` class and data preprocessing functions.

- **Key Features**:
  - Handles multiple datasets like AAFD and AFAD-Full.
  - Automatically reads CSV files to extract image paths and labels.
  - Includes data augmentation techniques such as resizing, flipping, and normalization.

- **Functions and Classes**:
  - `AgeGenderDataset`: A PyTorch `Dataset` class for loading and preprocessing images.
  - Preprocessing steps are defined using PyTorch's `transforms`.

---

### 2. `model.py`

Defines the `TimmAgeGenderModel` class, which is a neural network model for predicting gender and age.

- **Key Features**:
  - Uses a pre-trained backbone (e.g., MobileNet) from the `timm` library.
  - Two separate heads:
    - Gender classification head (Cross-Entropy Loss).
    - Age regression head (Mean Squared Error Loss).
  - Dropout layers are added to improve generalization.

- **Functions and Methods**:
  - `save_checkpoint`: Saves the model, optimizer, and scaler states.
  - `load_checkpoint`: Loads a previously saved checkpoint.

---

### 3. `train.py`

This file contains the main training pipeline.

- **Key Features**:
  - Loads the dataset using `AgeGenderDataset`.
  - Implements a training loop with mixed precision (`torch.cuda.amp`).
  - Logs training metrics (loss, accuracy) to TensorBoard.
  - Saves model checkpoints after each epoch.

- **Important Functions**:
  - `train_model`: The main training loop, which iterates over epochs and updates the model.
  - Integrated with `tqdm` for real-time progress display.

---

### 4. `utils.py`

Provides helper functions to simplify various tasks.

- **Key Features**:
  - Functions for preprocessing, such as resizing and normalizing images.
  - Additional utilities for handling paths, logging, or metrics.

---

### 5. `testCUDA.py`

A small script to test CUDA availability and GPU configuration.

- **Key Features**:
  - Prints available CUDA devices and their properties.
  - Verifies that PyTorch can successfully utilize GPU resources.

---

## Workflow

1. **Dataset Preparation**:
   - Organize your dataset and annotations in CSV files.
   - Use `dataset.py` to load and preprocess the data.

2. **Training**:
   - Run `train.py` to train the model. Checkpoints and logs are saved automatically.

3. **Evaluation**:
   - Use `model.py` to load saved checkpoints and perform inference.

4. **Customization**:
   - Modify `model.py` to experiment with different backbone architectures or hyperparameters.

---

## Example Commands

### Training
```bash
python train.py
