# Crop Type Segmentation with UNet 🌾🚜 Africa: Ghana Dataset

## Overview

This repository contains code for training a UNet-based model for crop type segmentation using satellite imagery. The model is implemented in PyTorch, and experiment tracking is performed using Weights & Biases (wandb).

## 1. Requirements

- Python 3.x
- PyTorch
- wandb (Weights & Biases)
- NumPy
- scikit-learn
- tqdm

Install the required packages using the following command:

```bash
pip install -r requirements.txt
```
## 2. Dataset
The dataset used for training is the SustainBenchDataset. Ensure that you have the dataset available or modify the code to load your dataset.

## 3. Model Architecture
The implemented model is a 2D UNet with multiple encoders for different satellite data sources (S2, S1, and Planet). The UNet architecture is defined in the `UNet_planet` class in the `2D_UNET_Model.py` file.

## 4. Training

Training is performed in the `train.py` script. Key configurations such as learning rate, loss weights, and epochs can be adjusted in the `config` dictionary.

To start training, run:

`python train.py`

The training progress and metrics are logged to Weights & Biases for easy tracking.

## 5. Custom Loss Function

The loss function used is a custom weighted cross-entropy loss (`mask_ce_loss`). Ensure that the loss function is suitable for your segmentation task.

## 6. Checkpoints

The best-performing model is saved during training based on the validation F1 score. Checkpoints are stored in the `/content/models/` directory.

## 3. Monitoring with wandb

The training and validation metrics are monitored using wandb. You can view the training progress and metrics on the wandb dashboard.

To log in and view the dashboard, run:

`wandb login`

Important Note:

This is a general overview of the Unet of 2D process.

## 4. Contributors

- **Deo Uwimpuhwe**
  - Program: MECE 2024
  - Email: [duwimpuh@andrew.cmu.edu](mailto:duwimpuh@andrew.cmu.edu)

- **Gustave Bwirayesu**
  - Program: MECE 2024
  - Email: [gbwiraye@andrew.cmu.edu](mailto:gbwiraye@andrew.cmu.edu)

- **Eric Maniraguha**
  - Program: MSIT 2024
  - Email: [emanirag@andrew.cmu.edu](mailto:emanirag@andrew.cmu.edu)

- **Bienvenue Jules Himbaza**
  - Program: MECE 2024
  - Email: [hjulesbi@andrew.cmu.edu](mailto:hjulesbi@andrew.cmu.edu)


## Remind ourself Deep Learning 🤖🎉

- **Gradient Descent Dance**: Watch as your model grooves its way down the loss function!
  
  ![Gradient Descent Dance](icons/dance.gif)

- **Backpropagation Boogie**: The funky steps your gradients take to update those weights!

  ![Backpropagation Boogie](icons/boogie.gif)

- **Epoch Extravaganza**: Each epoch is a party for your neural network!

  ![Epoch Extravaganza](icons/party.gif)

- **Loss Limbo**: How low can you go? Challenge your model to limbo under the loss bar!

  ![Loss Limbo](icons/limbo.gif)

...


