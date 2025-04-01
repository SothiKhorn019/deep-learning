#!/usr/bin/env python3
import argparse
import datetime
import os
import re

import numpy as np
import timm
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torchvision.transforms.v2 as v2
import torchmetrics

import npfl138
npfl138.require_version("2425.5")
from npfl138.datasets.cags import CAGS

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=32, type=int, help="Batch size.")
parser.add_argument("--epochs", default=3, type=int, help="Number of epochs for phase 1.")
parser.add_argument("--finetune_epochs", default=20, type=int, help="Number of epochs for phase 2 fine-tuning.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")

def main(args: argparse.Namespace) -> None:
    npfl138.startup(args.seed, args.threads)
    npfl138.global_keras_initializers()
    
    args.logdir = os.path.join("logs", "{}-{}-{}".format(
        os.path.basename(__file__),
        datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        ",".join("{}={}".format(k, v) for k, v in sorted(vars(args).items()))
    ))
    
    # Load the data.
    cags = CAGS(decode_on_demand=False)
    
    # Load the pretrained EfficientNetV2-B0 without its classifier.
    efficientnetv2_b0 = timm.create_model("tf_efficientnetv2_b0.in1k", pretrained=True, num_classes=0)
    
    # Define strong augmentation for training.
    train_preprocessing = v2.Compose([
        v2.ToDtype(torch.float32, scale=True),  # Converts to float and scales to [0,1]
        v2.Resize(256),
        v2.RandomCrop(224),
        v2.RandomHorizontalFlip(),
        v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        v2.Normalize(mean=efficientnetv2_b0.pretrained_cfg["mean"],
                     std=efficientnetv2_b0.pretrained_cfg["std"]),
    ])
    
    # Evaluation pipeline.
    eval_preprocessing = v2.Compose([
        v2.ToDtype(torch.float32, scale=True),
        v2.Resize(224),
        v2.Normalize(mean=efficientnetv2_b0.pretrained_cfg["mean"],
                     std=efficientnetv2_b0.pretrained_cfg["std"]),
    ])
    
    # Define collate functions.
    def train_collate_fn(batch):
        images = [train_preprocessing(example["image"]) for example in batch]
        labels = [example["label"] for example in batch]
        return torch.stack(images), torch.tensor(labels)
    
    def eval_collate_fn(batch):
        images = [eval_preprocessing(example["image"]) for example in batch]
        labels = [example["label"] for example in batch]
        return torch.stack(images), torch.tensor(labels)
    
    train_loader = torch.utils.data.DataLoader(cags.train, batch_size=args.batch_size,
                                                shuffle=True, collate_fn=train_collate_fn)
    dev_loader   = torch.utils.data.DataLoader(cags.dev, batch_size=args.batch_size,
                                                collate_fn=eval_collate_fn)
    test_loader  = torch.utils.data.DataLoader(cags.test, batch_size=args.batch_size,
                                                collate_fn=eval_collate_fn)
    
    # Define a classifier model using the pretrained feature extractor.
    class CAGSClassifier(nn.Module):
        def __init__(self, feature_extractor, num_classes):
            super().__init__()
            self.feature_extractor = feature_extractor
            self.classifier = nn.Linear(1280, num_classes)
            # Initially freeze the feature extractor.
            for param in self.feature_extractor.parameters():
                param.requires_grad = False
                
        def forward(self, x):
            features = self.feature_extractor(x)  # shape: [batch, 1280]
            return self.classifier(features)
    
    model = CAGSClassifier(efficientnetv2_b0, CAGS.LABELS)
    trainable_model = npfl138.TrainableModule(model)
    
    # Phase 1: Train only the classifier head.
    optimizer = optim.Adam(trainable_model.module.classifier.parameters(), lr=0.001)
    trainable_model.configure(
        optimizer=optimizer,
        loss=nn.CrossEntropyLoss(),
        metrics={"accuracy": torchmetrics.Accuracy(task="multiclass", num_classes=CAGS.LABELS)},
        logdir=args.logdir,
    )
    
    print("Phase 1: Training classifier head...")
    trainable_model.fit(train_loader, dev=dev_loader, epochs=args.epochs)
    
    # Phase 2: Fine-tune the entire model.
    print("Phase 2: Fine-tuning entire model...")
    # Unfreeze the feature extractor.
    for param in trainable_model.module.feature_extractor.parameters():
        param.requires_grad = True
        
    optimizer_ft = optim.Adam(trainable_model.parameters(), lr=0.0001, weight_decay=1e-4)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer_ft, T_max=args.finetune_epochs)
    trainable_model.configure(
        optimizer=optimizer_ft,
        loss=nn.CrossEntropyLoss(),
        metrics={"accuracy": torchmetrics.Accuracy(task="multiclass", num_classes=CAGS.LABELS)},
        logdir=args.logdir,
    )
    
    for epoch in range(args.finetune_epochs):
        print(f"Fine-tuning Epoch {epoch+1}/{args.finetune_epochs}")
        trainable_model.fit(train_loader, dev=dev_loader, epochs=1)
        scheduler.step()
    
    # Generate test set annotations.
    os.makedirs(args.logdir, exist_ok=True)
    with open(os.path.join(args.logdir, "cags_classification.txt"), "w", encoding="utf-8") as f:
        for prediction in trainable_model.predict(test_loader, data_with_labels=True):
            print(np.argmax(prediction), file=f)
            
if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    main(main_args)
