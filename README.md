# MNIST CNN with Weights & Biases

A PyTorch implementation of a CNN for MNIST digit classification with Weights & Biases experiment tracking and a FastAPI inference server.

## Features

- Simple CNN architecture for MNIST digit classification
- Weights & Biases integration for experiment tracking
- FastAPI-based REST API for model inference
- Web interface for digit prediction

## Project Structure

```
.
├── model.py       # SimpleCNN model definition
├── train.py       # Training script with wandb logging
├── predict.py     # Predictor class for inference
├── api.py         # FastAPI server
├── index.html     # Web UI for predictions
├── utils.py       # Utility functions
└── model.pth      # Trained model weights
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Training

Train the model with default parameters:

```bash
python train.py
```

Available options:
- `--lr`: Learning rate (default: 1e-3)
- `--batch-size`: Batch size (default: 64)
- `--epochs`: Number of epochs (default: 5)
- `--project`: Weights & Biases project name (default: wandb-mnist-demo)
- `--wandb-mode`: wandb mode - online/offline/disabled (default: online)
- `--log-samples`: Log sample predictions to wandb

Example:

```bash
python train.py --epochs 10 --lr 0.001 --batch-size 128 --log-samples
```

### Inference API

Start the FastAPI server:

```bash
python api.py
```

The server runs at `http://localhost:8000` with:
- `GET /`: Web interface for uploading images
- `POST /predict`: Prediction endpoint

API example:

```bash
curl -X POST "http://localhost:8000/predict" \
  -F "image=@digit.png"
```

Response:

```json
{
  "prediction": 7,
  "probabilities": [0.001, 0.002, ..., 0.95, ...]
}
```

## Model Architecture

SimpleCNN:
- Conv2d(1, 32, 3) + ReLU + MaxPool(2)
- Conv2d(32, 64, 3) + ReLU + MaxPool(2)
- Linear(3136, 128) + ReLU
- Linear(128, 10)

## Requirements

- PyTorch
- torchvision
- wandb
- fastapi[standard]
