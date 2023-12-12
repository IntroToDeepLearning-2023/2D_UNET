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

### Clone Repository

Use the following command to clone the repository:

```bash
git clone https://github.com/IntroToDeepLearning-2023/2D_UNET.git

Install the required packages using the following command:

```bash
pip install -r requirements.txt
```
## 2. Dataset
The dataset used for training is the SustainBenchDataset. Ensure that you have the dataset available or modify the code to load your dataset.

### Accessing Data on GCP

To access the dataset on GCP, follow these steps:

1. **GCP Account Setup:**
   - Ensure you have access to the Google Cloud Platform with the necessary permissions.

2. **Dataset Location:**
   - The dataset is stored in a GCP Storage Bucket. You can find the dataset at the following location: `gs://your-bucket/dataset-folder`.

3. **Downloading the Dataset:**
   - Use the following command to list files in the dataset folder:

     ```bash
     !gsutil ls gs://data_ctm/data/data/africa_crop_type_mapping/ghana/
     ```

     You can download specific files using `gsutil cp` command.

     ```bash
     !gsutil cp gs://data_ctm/data/data/africa_crop_type_mapping/ghana/your_file.csv .
     ```

## 3. Model Architecture
The implemented model is a 2D UNet with multiple encoders for different satellite data sources (S2, S1, and Planet). The UNet architecture is defined in the `UNet_planet` class in the `2D_UNET_Model.py` file.

## 4. Training

*1. Configure Parameters:* Adjust key configurations in the `config` dictionary in the `train.py` script. You can set the learning rate, loss weights, epochs, etc.

*2. Start Training:* Run the following command to start training:

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

## 4. Inference

After training, you can use the trained model for inference on new data. Modify the `inference.py` script or create a new script for your specific use case.

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


