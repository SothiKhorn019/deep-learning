#!/usr/bin/env python3
# Group IDs: 31ff17c9-b0b8-449e-b0ef-8a1aa1e14eb3, 5b78caaa-8040-46f7-bf54-c13e183bbbf8

import argparse
import datetime
import os
import re

import torch
import timm
import numpy as np
import torchvision.transforms.v2 as v2
import torchvision.ops as ops

import bboxes_utils
import npfl138
from npfl138.datasets.svhn import SVHN

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=32, type=int)
parser.add_argument("--epochs", default=11, type=int)
parser.add_argument("--seed", default=42, type=int)
parser.add_argument("--threads", default=0, type=int)
parser.add_argument("--lr", default=5e-4, type=float)
parser.add_argument("--weight_decay", default=1e-4, type=float)


class SVHNDataset(npfl138.TransformedDataset):
    def __init__(self, dataset, transform):
        super().__init__(dataset)
        self._transform = transform

    def transform(self, example):
        orig_h, orig_w = example["image"].shape[1:]
        image = self._transform(example["image"])
        new_h, new_w = image.shape[1:]

        bboxes = example["bboxes"].clone().float()
        bboxes[:, [0, 2]] *= new_h / orig_h
        bboxes[:, [1, 3]] *= new_w / orig_w

        return image, example["classes"], bboxes, (orig_h, orig_w)


def detection_collate_fn(batch):
    images = torch.stack([x[0] for x in batch])
    sizes = torch.tensor([x[3] for x in batch])

    max_len = max(len(x[1]) for x in batch)
    padded_classes = torch.full((len(batch), max_len), -1, dtype=torch.long)
    padded_bboxes = torch.zeros((len(batch), max_len, 4))

    for i, (cls, box) in enumerate([(x[1], x[2]) for x in batch]):
        padded_classes[i, :len(cls)] = cls
        padded_bboxes[i, :len(box)] = box

    return (images, sizes), (padded_classes, padded_bboxes)


def generate_anchors(image_size=224, feature_size=7, box_size=32):
    step = image_size // feature_size
    anchors = [
        [step * (y + 0.5) - box_size / 2, step * (x + 0.5) - box_size / 2,
         step * (y + 0.5) + box_size / 2, step * (x + 0.5) + box_size / 2]
        for y in range(feature_size) for x in range(feature_size)
    ]
    return torch.tensor(anchors, dtype=torch.float32)


class RetinaNet(npfl138.TrainableModule):
    def __init__(self, backbone, num_classes, device, freeze_backbone=True):
        super().__init__()
        self.device = device
        self.num_classes = num_classes
        self.backbone = backbone
        self.out_channels = self.backbone.feature_info[-1]['num_chs']
        self.feature_size = 256
        self.anchors = generate_anchors().to(self.device)

        self.fpn = torch.nn.Sequential(
            torch.nn.Conv2d(self.out_channels, self.feature_size, 1),
            torch.nn.Conv2d(self.feature_size, self.feature_size, 3, padding=1)
        )

        def head(out_dim):
            return torch.nn.Sequential(
                torch.nn.Conv2d(self.feature_size, self.feature_size, 3, padding=1),
                torch.nn.ReLU(),
                torch.nn.Conv2d(self.feature_size, self.feature_size, 3, padding=1),
                torch.nn.ReLU(),
                torch.nn.Conv2d(self.feature_size, out_dim, 3, padding=1),
            )

        self.cls_head = head(self.num_classes + 1)
        self.box_head = head(4)

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def forward(self, x, sizes=None):
        C5 = self.backbone(x)[-1]
        P5 = self.fpn(C5)

        cls_logits = self.cls_head(P5).permute(0, 2, 3, 1).reshape(x.size(0), -1, self.num_classes + 1)
        box_deltas = self.box_head(P5).permute(0, 2, 3, 1).reshape(x.size(0), -1, 4)
        return (cls_logits, box_deltas), self.anchors

    def compute_loss(self, y_pred, y, *_):
        (cls_logits, box_deltas), anchors = y_pred
        gold_classes, gold_bboxes = y
        anchor_classes_all, anchor_bboxes_all = [], []

        for cls, box in zip(gold_classes, gold_bboxes):
            acls, abox = bboxes_utils.bboxes_training(anchors, cls, box, iou_threshold=0.5)
            anchor_classes_all.append(acls)
            anchor_bboxes_all.append(abox)

        anchor_classes = torch.stack(anchor_classes_all)
        anchor_bboxes = torch.stack(anchor_bboxes_all)

        target_cls = torch.nn.functional.one_hot(anchor_classes, self.num_classes + 1).float()
        cls_loss = ops.sigmoid_focal_loss(cls_logits, target_cls, reduction="mean")

        positive = anchor_classes > 0
        reg_loss = torch.nn.functional.smooth_l1_loss(
            box_deltas[positive], anchor_bboxes[positive], reduction="mean"
        ) if positive.any() else torch.tensor(0.0, device=self.device)

        return cls_loss + reg_loss

    def predict(self, images, score_thresh=0.5, iou_thresh=0.5):
        self.eval()
        with torch.no_grad():
            (cls_logits, box_deltas), anchors = self(images)
            probs = torch.sigmoid(cls_logits)
            results = []
            for i in range(images.size(0)):
                scores, labels = probs[i][:, 1:].max(-1)
                keep = scores > score_thresh
                if keep.any():
                    boxes = bboxes_utils.bboxes_from_rcnn(anchors, box_deltas[i])[keep]
                    keep_nms = ops.batched_nms(boxes, scores[keep], labels[keep], iou_thresh)
                    results.append((labels[keep][keep_nms], boxes[keep_nms]))
                else:
                    results.append((torch.empty(0, dtype=torch.long), torch.empty(0, 4)))
            return results

    def train(self, mode=True):
        self.backbone.eval()
        self.cls_head.train(mode)
        self.box_head.train(mode)
        return self


def prepare_dataloaders(svhn, train_transform, eval_transform, batch_size):
    # Use separate transforms for training and evaluation
    loader = lambda ds, transform, shuffle: torch.utils.data.DataLoader(
        SVHNDataset(ds, transform),
        batch_size=batch_size,
        collate_fn=detection_collate_fn,
        num_workers=0,
        shuffle=shuffle
    )
    train_loader = loader(svhn.train, train_transform, True)
    dev_loader = loader(svhn.dev, eval_transform, True)
    dev_test_loader = loader(svhn.dev, eval_transform, False)
    test_loader = loader(svhn.test, eval_transform, False)
    return train_loader, dev_loader, dev_test_loader, test_loader


def main(args):
    npfl138.startup(args.seed, args.threads)
    npfl138.global_keras_initializers()

    args.logdir = os.path.join("logs", f"RetinaNet-{datetime.datetime.now():%Y-%m-%d_%H%M%S}")

    svhn = SVHN(decode_on_demand=False)
    efficientnet = timm.create_model("tf_efficientnetv2_b0.in1k", pretrained=True, features_only=True)

    # Define separate transform pipelines:
    # Training transform with color jitter augmentation.
    train_transform = v2.Compose([
        v2.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        v2.Resize((224, 224)),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=efficientnet.pretrained_cfg["mean"], std=efficientnet.pretrained_cfg["std"]),
    ])
    # Evaluation transform (no augmentation).
    eval_transform = v2.Compose([
        v2.Resize((224, 224)),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=efficientnet.pretrained_cfg["mean"], std=efficientnet.pretrained_cfg["std"]),
    ])

    # Use the new transforms for dataloader preparation.
    train, dev, dev_test, test = prepare_dataloaders(svhn, train_transform, eval_transform, args.batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RetinaNet(efficientnet, svhn.LABELS, device).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    model.configure(optimizer=optimizer, loss=model.compute_loss, logdir=args.logdir)
    model.fit(train, dev=dev, epochs=args.epochs)

    with open(os.path.join(args.logdir, "svhn_competition.txt"), "w") as predictions_file:
        for (images, sizes), _ in test:
            images, sizes = images.to(device), sizes.to(device)
            preds = model.predict(images, score_thresh=0.25)
            for i, (predicted_classes, predicted_bboxes) in enumerate(preds):
                # Scale predicted bounding boxes from 224x224 back to original image size.
                if predicted_bboxes.numel():
                    predicted_bboxes[:, [0, 2]] *= sizes[i][0] / 224.0
                    predicted_bboxes[:, [1, 3]] *= sizes[i][1] / 224.0

                output = []
                for label, bbox in zip(predicted_classes, predicted_bboxes):
                    # Append the label (converted to int) and bounding box coordinates as floats.
                    output += [int(label.item())] + list(map(float, bbox))
                print(*output, file=predictions_file)


    # print("Visualize with:", f"python3 -m npfl138.datasets.svhn --visualize={args.logdir}/svhn_competition.txt --dataset=dev")


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
