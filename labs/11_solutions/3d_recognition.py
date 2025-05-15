#!/usr/bin/env python3
import argparse
import datetime
import os
import re
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import npfl138
npfl138.require_version("2425.11")
from npfl138.datasets.modelnet import ModelNet

# Define command-line arguments with sensible defaults
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=32, type=int, help="Batch size.")
parser.add_argument("--epochs", default=20, type=int, help="Number of epochs.")
parser.add_argument(
    "--modelnet", default=20, type=int, choices=[20, 32],
    help="ModelNet resolution (20 or 32)."
)
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument(
    "--threads", default=0, type=int,
    help="Number of CPU threads to use (0 = all available cores)."
)

class Net(nn.Module):
    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(32)
        self.conv2 = nn.Conv3d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(64)
        self.conv3 = nn.Conv3d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm3d(128)
        self.pool = nn.MaxPool3d(2)
        self.avgpool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


def main(args: argparse.Namespace) -> None:
    # Initialize seed and threads
    npfl138.startup(args.seed, args.threads)
    npfl138.global_keras_initializers()

    # Create logdir name.
    args.logdir = os.path.join("logs", "{}-{}-{}".format(
        os.path.basename(globals().get("__file__", "notebook")),
        datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", k), v) for k, v in sorted(vars(args).items())))
    ))
    os.makedirs(args.logdir, exist_ok=True)

    # Load ModelNet datasets
    modelnet = ModelNet(args.modelnet)
    train_ds, dev_ds, test_ds = modelnet.train, modelnet.dev, modelnet.test
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    dev_loader   = DataLoader(dev_ds,   batch_size=args.batch_size)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size)

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Instantiate model, loss, and optimizer
    model = Net(in_channels=ModelNet.C, num_classes=ModelNet.LABELS).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    best_dev_acc = 0.0
    best_model_path = os.path.join(args.logdir, "best_model.pt")

    # Training loop
    for epoch in range(1, args.epochs + 1):
        model.train()
        for sample in train_loader:
            grids = sample["grid"].to(device).float()
            labels = sample["label"].to(device).long()
            optimizer.zero_grad()
            outputs = model(grids)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        # Evaluate on development set
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for sample in dev_loader:
                grids = sample["grid"].to(device).float()
                labels = sample["label"].to(device).long()
                preds = model(grids).argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        dev_acc = 100 * correct / total
        print(f"Epoch {epoch}/{args.epochs}, Dev Acc: {dev_acc:.2f}%")

        # Save best model
        if dev_acc > best_dev_acc:
            best_dev_acc = dev_acc
            torch.save(model.state_dict(), best_model_path)

    print(f"Best Dev Acc: {best_dev_acc:.2f}%, saved at {best_model_path}")

    # Load best model and generate test predictions
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()
    preds_list = []
    with torch.no_grad():
        for sample in test_loader:
            grids = sample["grid"].to(device).float()
            preds = model(grids).argmax(dim=1).cpu().numpy()
            preds_list.extend(preds.tolist())

    # Save test predictions
    out_path = os.path.join(args.logdir, "3d_recognition.txt")
    with open(out_path, "w", encoding="utf-8") as out:
        for p in preds_list:
            print(p, file=out)

if __name__ == "__main__":
    main(parser.parse_args())
