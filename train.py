import argparse
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import wandb
from utils import get_device
from model import SimpleCNN



def get_dataloaders(batch_size: int) -> Tuple[DataLoader, DataLoader]:
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    train_dataset = datasets.MNIST(
        root="data", train=True, download=True, transform=transform
    )
    val_dataset = datasets.MNIST(
        root="data", train=False, download=True, transform=transform
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


def log_sample_predictions(
    images: torch.Tensor,
    labels: torch.Tensor,
    preds: torch.Tensor,
) -> List[wandb.Image]:
    # wandb.Image does not support batched tensors, so log individual examples.
    max_samples = 8
    images = images[:max_samples]
    labels = labels[:max_samples]
    preds = preds[:max_samples]

    logged_images = []
    for img, label, pred in zip(images, labels, preds):
        img_2d = img.squeeze(0)  # remove single-channel dimension
        logged_images.append(
            wandb.Image(
                img_2d.numpy(),
                caption=f"Pred: {pred.item()} | Label: {label.item()}",
            )
        )
    return logged_images


def train(config):
    device = get_device()
    print(f"Using device: {device}")
    wandb.init(project=config.project, config=config, mode=config.wandb_mode)
    model = SimpleCNN().to(device)
    wandb.watch(model, log="gradients", log_freq=100)

    train_loader, val_loader = get_dataloaders(config.batch_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.lr)

    for epoch in range(1, config.epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (images, labels) in enumerate(train_loader, start=1):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_loss = running_loss / total
        train_acc = correct / total

        val_loss, val_acc = evaluate(model, val_loader, device)

        log_data = {
            "epoch": epoch,
            "train/loss": train_loss,
            "train/accuracy": train_acc,
            "val/loss": val_loss,
            "val/accuracy": val_acc,
        }

        if config.log_samples:
            sample_images, sample_labels = next(iter(val_loader))
            sample_images = sample_images.to(device)
            with torch.no_grad():
                preds = model(sample_images).argmax(dim=1)
            log_data["val/sample_predictions"] = log_sample_predictions(
                sample_images[:16].cpu(), sample_labels[:16], preds[:16].cpu()
            )

        wandb.log(log_data)
        print(
            f"Epoch {epoch}/{config.epochs} "
            f"- train_loss: {train_loss:.4f}, train_acc: {train_acc:.4f} "
            f"- val_loss: {val_loss:.4f}, val_acc: {val_acc:.4f}"
        )

    wandb.finish()
    return model


def parse_args():
    parser = argparse.ArgumentParser(description="Simple MNIST CNN with wandb logging")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument(
        "--project",
        type=str,
        default="wandb-mnist-demo",
        help="wandb project name",
    )
    parser.add_argument(
        "--wandb-mode",
        type=str,
        default="online",
        choices=["online", "offline", "disabled"],
        help="wandb init mode",
    )
    parser.add_argument(
        "--log-samples",
        action="store_true",
        help="Log a batch of sample predictions/images",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    model = train(args)
    torch.save(model.state_dict(), "model.pth") 
    
