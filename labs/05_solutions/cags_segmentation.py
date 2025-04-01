#!/usr/bin/env python3
import argparse
import datetime
import os
import re

import numpy as np
import timm
import torch
import torch.nn as nn
import torchvision.transforms.v2 as v2

import npfl138
npfl138.require_version("2425.5")
from npfl138.datasets.cags import CAGS

# TODO: Define reasonable defaults and optionally more parameters.
# Also, you can set the number of threads to 0 to use all your CPU cores.
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=8, type=int, help="Batch size.")
parser.add_argument("--epochs", default=30, type=int, help="Number of epochs.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")


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

    # Load the data. The individual examples are dictionaries with the keys:
    # - "image", a `[3, 224, 224]` tensor of `torch.uint8` values in [0-255] range,
    # - "mask", a `[1, 224, 224]` tensor of `torch.float32` values in [0-1] range,
    # - "label", a scalar of the correct class in `range(CAGS.LABELS)`.
    # The `decode_on_demand` argument can be set to `True` to save memory and decode
    # each image only when accessed, but it will most likely slow down training.
    cags = CAGS(decode_on_demand=False)

    # Load the EfficientNetV2-B0 model without the classification layer.
    # Apart from calling the model as in the classification task, you can call it using
    #   output, features = efficientnetv2_b0.forward_intermediates(batch_of_images)
    # obtaining (assuming the input images have 224x224 resolution):
    # - `output` is a `[N, 1280, 7, 7]` tensor with the final features before global average pooling,
    # - `features` is a list of intermediate features with resolution 112x112, 56x56, 28x28, 14x14, 7x7.
    efficientnetv2_b0 = timm.create_model("tf_efficientnetv2_b0.in1k", pretrained=True, num_classes=0)

    # Create a simple preprocessing performing necessary normalization.
    preprocessing = v2.Compose([
        v2.ToDtype(torch.float32, scale=True),  # The `scale=True` also rescales the image to [0, 1].
        v2.Normalize(mean=efficientnetv2_b0.pretrained_cfg["mean"], std=efficientnetv2_b0.pretrained_cfg["std"]),
    ])

    # TODO: Create the model and train it.
    # Define the segmentation model.
    class CAGSSegmentationModel(nn.Module):
        def __init__(self, backbone):
            super().__init__()
            self.backbone = backbone  # EfficientNetV2-B0

            # Decoder: Upsample from [N,1280,7,7] to [N,1,224,224]
            self.up1 = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                nn.Conv2d(1280, 512, kernel_size=3, padding=1),
                nn.ReLU(inplace=True)
            )  # Output: [N,512,14,14]
            self.up2 = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                nn.Conv2d(512, 256, kernel_size=3, padding=1),
                nn.ReLU(inplace=True)
            )  # Output: [N,256,28,28]
            self.up3 = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                nn.Conv2d(256, 128, kernel_size=3, padding=1),
                nn.ReLU(inplace=True)
            )  # Output: [N,128,56,56]
            self.up4 = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                nn.Conv2d(128, 64, kernel_size=3, padding=1),
                nn.ReLU(inplace=True)
            )  # Output: [N,64,112,112]
            self.up5 = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                nn.Conv2d(64, 32, kernel_size=3, padding=1),
                nn.ReLU(inplace=True)
            )  # Output: [N,32,224,224]
            self.final_conv = nn.Conv2d(32, 1, kernel_size=1)  # Output: [N,1,224,224]

        def forward(self, x):
            # x: [N,3,224,224]
            output, features = self.backbone.forward_intermediates(x)
            # Use the final feature map (shape: [N,1280,7,7])
            x = output
            x = self.up1(x)  # [N,512,14,14]
            x = self.up2(x)  # [N,256,28,28]
            x = self.up3(x)  # [N,128,56,56]
            x = self.up4(x)  # [N,64,112,112]
            x = self.up5(x)  # [N,32,224,224]
            x = self.final_conv(x)  # [N,1,224,224]
            return torch.sigmoid(x)  # Produce probabilities in [0,1]

    model = CAGSSegmentationModel(efficientnetv2_b0)
    model = npfl138.TrainableModule(model)
    
    # Create a collate function that applies preprocessing.
    def collate_fn(batch):
        images = [preprocessing(example["image"]) for example in batch]
        masks = [example["mask"].float() for example in batch]
        return torch.stack(images), torch.stack(masks)

    train_loader = torch.utils.data.DataLoader(cags.train, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    dev_loader   = torch.utils.data.DataLoader(cags.dev, batch_size=args.batch_size, collate_fn=collate_fn)
    test_loader  = torch.utils.data.DataLoader(cags.test, batch_size=args.batch_size, collate_fn=collate_fn)

    # Configure the model: use BCELoss (since the output is sigmoid) and the MaskIoUMetric.
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    model.configure(
        optimizer=optimizer,
        loss=torch.nn.BCELoss(),
        metrics={"iou": CAGS.MaskIoUMetric()},
        logdir=args.logdir,
    )

    model.fit(train_loader, dev=dev_loader, epochs=args.epochs)

    # Generate test set annotations, but in `args.logdir` to allow parallel execution.
    os.makedirs(args.logdir, exist_ok=True)
    with open(os.path.join(args.logdir, "cags_segmentation.txt"), "w", encoding="utf-8") as predictions_file:
        # TODO: Perform the prediction on the test data. The line below assumes you have
        # a dataloader `test` where the individual examples are `(image, target)` pairs.
        for mask in model.predict(test_loader, data_with_labels=True):
            zeros, ones, runs = 0, 0, []
            for pixel in np.reshape(mask >= 0.5, [-1]):
                if pixel:
                    if zeros or (not zeros and not ones):
                        runs.append(zeros)
                        zeros = 0
                    ones += 1
                else:
                    if ones:
                        runs.append(ones)
                        ones = 0
                    zeros += 1
            runs.append(zeros + ones)
            print(*runs, file=predictions_file)


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    main(main_args)
