#!/usr/bin/env python3
import argparse
import datetime
import os
import re

import torch
import torchmetrics

import npfl138
npfl138.require_version("2425.4")
from npfl138.datasets.mnist import MNIST

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--batch_size", default=50, type=int, help="Batch size.")
parser.add_argument("--cnn", default=None, type=str, help="CNN architecture.")
parser.add_argument("--epochs", default=10, type=int, help="Number of epochs.")
parser.add_argument("--recodex", default=False, action="store_true", help="Evaluation in ReCodEx.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
# If you add more arguments, ReCodEx will keep them with your default values.


class Dataset(npfl138.TransformedDataset):
    def transform(self, example):
        image = example["image"]  # a torch.Tensor with torch.uint8 values in [0, 255] range
        image = image.to(torch.float32) / 255  # image converted to float32 and rescaled to [0, 1]
        label = example["label"]  # a torch.Tensor with a single integer representing the label
        return image, label  # return an (input, target) pair
    
class ResidualBlock(torch.nn.Module):
    def __init__(self, spec: str, in_channels: int):
        super().__init__()
        modules = []
        current_channels = in_channels
        for layer_spec in spec.split(','):
            parts = layer_spec.split('-')
            if parts[0] in ["C", "CB"]:
                filters = int(parts[1])
                kernel = int(parts[2])
                stride = int(parts[3])
                if parts[4] == "valid":
                    padding = 0
                elif parts[4] == "same":
                    padding = "same"
                if parts[0] == "C":
                    modules.append(torch.nn.Conv2d(current_channels, filters, kernel, stride, padding))
                    modules.append(torch.nn.ReLU())
                else: # parts[0] == "CB"
                    modules.append(torch.nn.Conv2d(current_channels, filters, kernel, stride, padding, bias=False))
                    modules.append(torch.nn.BatchNorm2d(filters))
                    modules.append(torch.nn.ReLU(inplace=True))
                current_channels = filters
                print()
            else:
                raise ValueError(f"Unsupported layer type in residual block: {parts[0]}")
        self.block = torch.nn.Sequential(*modules)

    def forward(self, x):
        return x + self.block(x)


class Model(npfl138.TrainableModule):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__()
        # TODO: Add CNN layers specified by `args.cnn`, which contains
        # a comma-separated list of the following layers:
        # - `C-filters-kernel_size-stride-padding`: Add a convolutional layer with ReLU
        #   activation and specified number of filters, kernel size, stride and padding.
        # - `CB-filters-kernel_size-stride-padding`: Same as `C`, but use batch normalization.
        #   In detail, start with a convolutional layer **without bias** and activation,
        #   then add a batch normalization layer, and finally the ReLU activation.
        # - `M-pool_size-stride`: Add max pooling with specified size and stride, using
        #   the default padding of 0 (the "valid" padding).
        # - `R-[layers]`: Add a residual connection. The `layers` contain a specification
        #   of at least one convolutional layer (but not a recursive residual connection `R`).
        #   The input to the `R` layer should be processed sequentially by `layers`, and the
        #   produced output (after the ReLU nonlinearity of the last layer) should be added
        #   to the input (of this `R` layer).
        # - `F`: Flatten inputs. Must appear exactly once in the architecture.
        # - `H-hidden_layer_size`: Add a dense layer with ReLU activation and the specified size.
        # - `D-dropout_rate`: Apply dropout with the given dropout rate.
        # You can assume the resulting network is valid; it is fine to crash if it is not.
        #
        # To implement the residual connections, you can use various approaches, for example:
        # - you can create a specialized `torch.nn.Module` subclass representing a residual
        #   connection that gets the inside layers as an argument, and implement its forward call.
        #   This allows you to have the whole network in a single `torch.nn.Sequential`.
        # - you could represent the model module as a `torch.nn.ModuleList` of `torch.nn.Sequential`s,
        #   each representing one user-specified layer, keep track of the positions of residual
        #   connections, and manually perform them in the forward pass.
        #
        # It might be difficult to compute the number of features after the `F` layer. You can
        # nevertheless use the `torch.nn.LazyLinear` and `torch.nn.LazyConv2d` layers, which
        # do not require the number of input features to be specified in the constructor.
        # However, after the whole model is constructed, you must call the model once on a dummy input
        # so that the number of features is computed and the model parameters are initialized.
        # To that end, you can use for example
        #   self.eval()(torch.zeros(1, MNIST.C, MNIST.H, MNIST.W))
        # where the `self.eval()` is necessary to avoid the batchnorms to update their running statistics.

        # TODO: Finally, add the final Linear output layer with `MNIST.LABELS` units.
        layers = []
        in_channels = MNIST.C  # initial number of channels for MNIST images
        
        def split_layers(cnn_str: str):
            tokens = []
            current_token = []
            bracket_level = 0
            for char in cnn_str:
                if char == '[':
                    bracket_level += 1
                elif char == ']':
                    bracket_level -= 1
                # Only split on comma if we're not inside brackets.
                if char == ',' and bracket_level == 0:
                    tokens.append(''.join(current_token).strip())
                    current_token = []
                else:
                    current_token.append(char)
            if current_token:
                tokens.append(''.join(current_token).strip())
            return tokens
        
        layer_specs = split_layers(args.cnn)
        
        for spec in layer_specs:
            spec = spec.strip()
            parts = spec.split('-')
            if parts[0] in ["C", "CB"]:
                filters = int(parts[1])
                kernel = int(parts[2])
                stride = int(parts[3])
                if parts[4] == "valid":
                    padding = 0
                elif parts[4] == "same":
                    padding = "same"
                if parts[0] == "C":
                    layers.append(torch.nn.Conv2d(in_channels, filters, kernel, stride, padding))
                    layers.append(torch.nn.ReLU())
                else:  # parts[0] == "CB"
                    layers.append(torch.nn.Conv2d(in_channels, filters, kernel, stride, padding, bias=False))
                    layers.append(torch.nn.BatchNorm2d(filters))
                    layers.append(torch.nn.ReLU())
                in_channels = filters  # update input channels for the next layer
            elif parts[0] == 'M':
                pool_size, stride = map(int, parts[1:])
                layers.append(torch.nn.MaxPool2d(pool_size, stride))
            elif parts[0] == 'R':
                # Extract inner specification inside the brackets.
                inner_spec = spec[spec.find('[') + 1: spec.find(']')]
                # print(inner_spec)
                # if inner_spec.endswith('sam'):
                #         inner_spec += 'e'
                print('inner_spec',inner_spec)
                layers.append(ResidualBlock(inner_spec, in_channels))
            elif parts[0] == 'F':
                layers.append(torch.nn.Flatten())
            elif parts[0] == 'H':
                hidden_size = int(parts[1])
                layers.append(torch.nn.LazyLinear(hidden_size))
                layers.append(torch.nn.ReLU())
            elif parts[0] == 'D':
                dropout_rate = float(parts[1])
                layers.append(torch.nn.Dropout(dropout_rate))
            else:
                raise ValueError(f"Unknown layer type: {parts[0]}")
        # Add the final output layer.
        layers.append(torch.nn.LazyLinear(MNIST.LABELS))
        
        self.net = torch.nn.Sequential(*layers)
        
        # Initialize lazy layers by doing a dummy forward pass.
        self.eval()
        dummy_input = torch.zeros(1, MNIST.C, MNIST.H, MNIST.W)
        self.net(dummy_input)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def main(args: argparse.Namespace) -> dict[str, float]:
    # Set the random seed and the number of threads.
    npfl138.startup(args.seed, args.threads)
    npfl138.global_keras_initializers()

    # Create logdir name.
    args.logdir = os.path.join("logs", "{}-{}-{}".format(
        os.path.basename(globals().get("__file__", "notebook")),
        datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", k), v) for k, v in sorted(vars(args).items())))
    ))

    # Load the data and create dataloaders.
    mnist = MNIST()
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    train = torch.utils.data.DataLoader(Dataset(mnist.train), batch_size=args.batch_size, shuffle=True)
    dev = torch.utils.data.DataLoader(Dataset(mnist.dev), batch_size=args.batch_size)

    # Create the model and train it
    model = Model(args).to(device)

    model.configure(
        optimizer=torch.optim.Adam(model.parameters()),
        loss=torch.nn.CrossEntropyLoss(),
        metrics={"accuracy": torchmetrics.Accuracy("multiclass", num_classes=MNIST.LABELS)},
        logdir=args.logdir,
    )

    logs = model.fit(train, dev=dev, epochs=args.epochs)

    # Return development metrics for ReCodEx to validate.
    return {metric: value for metric, value in logs.items() if metric.startswith("dev_")}


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    main(main_args)
