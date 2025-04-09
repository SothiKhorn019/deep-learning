#!/usr/bin/env python3
import argparse
import datetime
import os
import re

import numpy as np
import timm
import torch
import torch.nn as nn
import torchvision.ops as ops
import torchvision.transforms.v2 as v2

import bboxes_utils
from bboxes_utils import TOP, LEFT, BOTTOM, RIGHT
import npfl138
npfl138.require_version("2425.6.1")
from npfl138.datasets.svhn import SVHN

# TODO: Define reasonable defaults and optionally more parameters.
# Also, you can set the number of threads to 0 to use all your CPU cores.
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=32, type=int, help="Batch size.")
parser.add_argument("--epochs", default=1, type=int, help="Number of epochs.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")

# Global constants.
IMAGE_SIZE = 224
GRID_SIZE = 7          # feature map is 7x7
STRIDE = IMAGE_SIZE / GRID_SIZE  # 32.0
ANCHOR_SIZE = 32.0     # fixed anchor box size
NUM_ANCHORS = GRID_SIZE * GRID_SIZE  # 49 anchors per image
NUM_DET_CLASSES = 11   # 0: background, 1..10: digits (for digits 0-9)

def generate_anchors():
    """Generate fixed anchors for an image of IMAGE_SIZE x IMAGE_SIZE.
    Each anchor is defined as [top, left, bottom, right] in pixel coordinates.
    We use one anchor per grid cell, with center at (i+0.5)*stride and fixed size.
    """
    anchors = []
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            center_y = (i + 0.5) * STRIDE
            center_x = (j + 0.5) * STRIDE
            half = ANCHOR_SIZE / 2.0
            anchors.append([center_y - half, center_x - half, center_y + half, center_x + half])
    return torch.tensor(anchors, dtype=torch.float32)

# Detection model.
class SVHNDetector(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone  # EfficientNetV2-B0, pretrained, num_classes=0.
        # Detection heads, applied on the final feature map (shape: [N,1280,7,7]).
        self.cls_head = nn.Conv2d(1280, NUM_DET_CLASSES, kernel_size=3, padding=1)
        self.reg_head = nn.Conv2d(1280, 4, kernel_size=3, padding=1)
        
    def forward(self, x):
        # x: [N, 3, IMAGE_SIZE, IMAGE_SIZE]
        # features = self.backbone(x)  # expected shape: [N, 1280, 7, 7]
        features, _ = self.backbone.forward_intermediates(x)
        cls_logits = self.cls_head(features)  # [N, NUM_DET_CLASSES, 7, 7]
        bbox_offsets = self.reg_head(features)  # [N, 4, 7, 7]
        N = x.shape[0]
        # Flatten spatial dimensions: result shapes: [N,49,NUM_DET_CLASSES] and [N,49,4]
        cls_logits = cls_logits.view(N, NUM_DET_CLASSES, -1).permute(0, 2, 1)
        bbox_offsets = bbox_offsets.view(N, 4, -1).permute(0, 2, 1)
        
        # cls_logits = cls_logits.reshape(N, NUM_DET_CLASSES, -1).permute(0, 2, 1)
        # bbox_offsets = bbox_offsets.reshape(N, 4, -1).permute(0, 2, 1)          
        return cls_logits, bbox_offsets

# Wrap the detector in a TrainableModule.
class SVHNDetectorModule(npfl138.TrainableModule):
    def compute_loss(self, outputs, targets, *args):
        # outputs: (cls_logits, bbox_offsets)
        # targets: (target_classes, target_offsets)
        cls_logits, bbox_offsets = outputs  # shapes: [N,49,NUM_DET_CLASSES], [N,49,4]
        target_classes, target_offsets = targets  # shapes: [N,49], [N,49,4]
        # Classification loss: use cross entropy; ignore anchors with target -1 (if any)
        # cls_loss = torch.nn.functional.cross_entropy(cls_logits.view(-1, NUM_DET_CLASSES), target_classes.view(-1), ignore_index=-1)
        cls_loss = torch.nn.functional.cross_entropy(cls_logits.reshape(-1, NUM_DET_CLASSES), target_classes.reshape(-1), ignore_index=-1)
        
        # Regression loss: L1 loss, computed only on positive anchors (target > 0)
        pos_mask = (target_classes > 0).unsqueeze(-1)  # [N,49,1]
        if pos_mask.sum() > 0:
            reg_loss = torch.nn.functional.l1_loss(bbox_offsets * pos_mask, target_offsets * pos_mask)
        else:
            reg_loss = 0.0
        return cls_loss + reg_loss

    def compute_metrics(self, outputs, targets, *args):
        cls_logits, _ = outputs
        target_classes, _ = targets
        preds = cls_logits.argmax(dim=-1)
        valid = (target_classes != -1)
        acc = (preds[valid] == target_classes[valid]).float().mean() if valid.sum() > 0 else torch.tensor(0.0)
        return {"accuracy": acc.item()}

# Generate training targets for one image.
def bboxes_target_for_image(anchors, gold_classes, gold_bboxes, iou_threshold=0.5):
    # gold_classes: [num_digits] (0..9), gold_bboxes: [num_digits,4]
    # bboxes_utils.bboxes_training expects gold_classes as tensor and returns
    # anchor_classes: [num_anchors] (0 for background, 1+digit for foreground)
    anchor_classes, anchor_offsets = bboxes_utils.bboxes_training(
        anchors, gold_classes, gold_bboxes, iou_threshold)
    return anchor_classes, anchor_offsets


# Prediction: decode one image.
def predict_one_image(model, image, anchors, conf_threshold=0.1, nms_iou_threshold=0.5):
    # image: [3, IMAGE_SIZE, IMAGE_SIZE], already preprocessed.
    image = image.unsqueeze(0)
    # model.eval()
    with torch.no_grad():
        cls_logits, bbox_offsets = model(image)
    cls_logits = cls_logits.squeeze(0)  # [49, NUM_DET_CLASSES]
    bbox_offsets = bbox_offsets.squeeze(0)  # [49,4]
    probs = torch.nn.functional.softmax(cls_logits, dim=-1)
    scores, labels = torch.max(probs, dim=-1)  # [49]
    # Filter out background (label==0) and low confidence.
    keep = (labels != 0) & (scores >= conf_threshold)
    if keep.sum() == 0:
        return [], []
    keep_idx = torch.nonzero(keep).squeeze(1)
    pred_labels = labels[keep_idx]
    pred_offsets = bbox_offsets[keep_idx]  # [M,4]
    selected_anchors = anchors[keep_idx]
    # Decode bbox using bboxes_from_rcnn.
    pred_bboxes = bboxes_utils.bboxes_from_rcnn(selected_anchors, pred_offsets)
    pred_scores = scores[keep_idx]
    # Apply NMS.
    keep_final = ops.nms(pred_bboxes, pred_scores, nms_iou_threshold)
    final_labels = pred_labels[keep_final]
    final_bboxes = pred_bboxes[keep_final]
    return final_labels.cpu().numpy(), final_bboxes.cpu().numpy()

def predict(model, data_loader):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    anchors = generate_anchors().to(device)  # fixed anchors
    predictions = []
    for images, _ in data_loader:
        for i in range(images.shape[0]):
            img = images[i]
            labels, bboxes = predict_one_image(model, img, anchors, conf_threshold=0.1, nms_iou_threshold=0.5)
            predictions.append((labels, bboxes))
    return predictions

def main(args: argparse.Namespace) -> None:
    # Set the random seed and the number of threads.
    npfl138.startup(args.seed, args.threads)
    npfl138.global_keras_initializers()

    # Create logdir name.
    args.logdir = os.path.join("logs", "{}-{}-{}".format(
        os.path.basename(globals().get("__file__", "notebook")),
        datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", k), v) for k, v in sorted(vars(args).items())))
    ))
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    # Load the data. The individual examples are dictionaries with the keys:
    # - "image", a `[3, SIZE, SIZE]` tensor of `torch.uint8` values in [0-255] range,
    # - "classes", a `[num_digits]` PyTorch vector with classes of image digits,
    # - "bboxes", a `[num_digits, 4]` PyTorch vector with bounding boxes of image digits.
    # The `decode_on_demand` argument can be set to `True` to save memory and decode
    # each image only when accessed, but it will most likely slow down training.
    svhn = SVHN(decode_on_demand=False)

    # Load the EfficientNetV2-B0 model without the classification layer.
    # Apart from calling the model as in the classification task, you can call it using
    #   output, features = efficientnetv2_b0.forward_intermediates(batch_of_images)
    # obtaining (assuming the input images have 224x224 resolution):
    # - `output` is a `[N, 1280, 7, 7]` tensor with the final features before global average pooling,
    # - `features` is a list of intermediate features with resolution 112x112, 56x56, 28x28, 14x14, 7x7.
    efficientnetv2_b0 = timm.create_model("tf_efficientnetv2_b0.in1k", pretrained=True, num_classes=0)

    # Create a simple preprocessing performing necessary normalization.
    preprocessing = v2.Compose([
        v2.Resize(224, interpolation=v2.InterpolationMode.BILINEAR),
        v2.ToDtype(torch.float32, scale=True),  # The `scale=True` also rescales the image to [0, 1].
        v2.Normalize(mean=efficientnetv2_b0.pretrained_cfg["mean"], std=efficientnetv2_b0.pretrained_cfg["std"]),
    ])
    
    # TODO: Create the model and train it.
    
    # Define the detector model.
    detector = SVHNDetector(efficientnetv2_b0)
    model = SVHNDetectorModule(detector)
    model.to(device)
    
    model.configure(
        optimizer=torch.optim.Adam(model.parameters(), lr=0.0001),
        loss=None,  # Loss is computed in compute_loss.
        metrics=None,
        logdir=args.logdir,
    )
    
    # Collate function for training.
    def collate_fn(batch):
        anchors = generate_anchors().to(device)  # fixed anchors for 224×224 images, shape [49,4]
        images = []
        target_cls = []
        target_offsets = []
        for elem in batch:
            # Get original image size.
            _, H, W = elem["image"].shape
            
            # Preprocess the image: Resize to 224×224, convert to float, scale and normalize.
            img = preprocessing(elem["image"])  # now shape [3,224,224]
            images.append(img)
            
            # Compute scaling factors from original size to 224.
            scale_y = 224 / H
            scale_x = 224 / W
            
            # Scale gold bounding boxes accordingly.
            gold_bboxes = elem["bboxes"].to(torch.float32).clone()  # shape [num_digits,4]
            gold_bboxes[:, TOP] *= scale_y
            gold_bboxes[:, BOTTOM] *= scale_y
            gold_bboxes[:, LEFT] *= scale_x
            gold_bboxes[:, RIGHT] *= scale_x
            
            gold_bboxes = gold_bboxes.to(device)
            gold_classes = elem["classes"].to(torch.long).to(device)
            
            # Generate training targets using the (resized) gold bboxes.
            a_cls, a_offsets = bboxes_target_for_image(anchors, gold_classes, gold_bboxes, 0.5)
            target_cls.append(a_cls)
            target_offsets.append(a_offsets)
        images = torch.stack(images).to(device)
        target_cls = torch.stack(target_cls).to(device)
        target_offsets = torch.stack(target_offsets).to(device)
        return images, (target_cls, target_offsets)
    
    train_loader = torch.utils.data.DataLoader(svhn.train, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    dev_loader = torch.utils.data.DataLoader(svhn.dev, batch_size=args.batch_size, collate_fn=collate_fn)
    test_loader = torch.utils.data.DataLoader(svhn.test, batch_size=args.batch_size, collate_fn=collate_fn)
    model.fit(train_loader, dev=dev_loader, epochs=args.epochs)
    
    
    dev_predictions = predict(model.module, dev_loader)
    dev_score = SVHN.evaluate(svhn.dev, dev_predictions, iou_threshold=0.5)
    print("Dev set IoU accuracy: {:.2f}%".format(dev_score))
    
    predictions = predict(model.module, test_loader)
    test_score = SVHN.evaluate(svhn.test, predictions, iou_threshold=0.5)
    print("Test set IoU accuracy: {:.2f}%".format(test_score))

    # Generate test set annotations, but in `args.logdir` to allow parallel execution.
    os.makedirs(args.logdir, exist_ok=True)
    with open(os.path.join(args.logdir, "svhn_competition.txt"), "w", encoding="utf-8") as predictions_file:
        # TODO: Predict the digits and their bounding boxes on the test set.
        # Assume that for a single test image we get
        # - `predicted_classes`: a 1D array with the predicted digits,
        # - `predicted_bboxes`: a [len(predicted_classes), 4] array with bboxes;
        for predicted_classes, predicted_bboxes in predict(model.module, test_loader):
            output = []
            for label, bbox in zip(predicted_classes, predicted_bboxes):
                output += [int(label)] + list(map(float, bbox))
            print(*output, file=predictions_file)


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    main(main_args)
