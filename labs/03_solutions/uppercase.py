#!/usr/bin/env python3
import argparse
import datetime
import os
import re

import torch
import torchmetrics
import numpy as np

import npfl138
npfl138.require_version("2425.3.1")
from npfl138.datasets.uppercase_data import UppercaseData

# TODO: Set reasonable values for the hyperparameters, especially for
# `alphabet_size`, `batch_size`, `epochs`, and `window`.
# Also, you can set the number of threads to 0 to use all your CPU cores.
parser = argparse.ArgumentParser()
parser.add_argument("--alphabet_size", default=100, type=int, help="If given, use this many most frequent chars.")
parser.add_argument("--batch_size", default=2048, type=int, help="Batch size.")
parser.add_argument("--epochs", default=40, type=int, help="Number of epochs.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=0, type=int, help="Maximum number of threads to use.")
parser.add_argument("--window", default=7, type=int, help="Window size to use.")

class BatchGenerator:
    """A simple batch generator, optionally with suffling.

    The functionality of this batch generator is very similar to
        torch.utils.data.DataLoader(
            torch.utils.data.StackDataset(inputs, outputs),
            batch_size=batch_size, shuffle=shuffle,
        )
    but if the data is stored in a single tensor, it is much faster.
    """
    def __init__(self, inputs: torch.Tensor, outputs: torch.Tensor, batch_size: int, shuffle: bool):
        self._inputs = inputs
        self._outputs = outputs
        self._batch_size = batch_size
        self._shuffle = shuffle

    def __len__(self):
        return (len(self._inputs) + self._batch_size - 1) // self._batch_size

    def __iter__(self):
        indices = torch.randperm(len(self._inputs)) if self._shuffle else torch.arange(len(self._inputs))
        while len(indices):
            batch = indices[:self._batch_size]
            indices = indices[self._batch_size:]
            yield self._inputs[batch], self._outputs[batch]


class Model(npfl138.TrainableModule):
    def __init__(self, args: argparse.Namespace):
        super().__init__()
        self._args = args
        
        # TODO: Implement a suitable model. The inputs are _windows_ of fixed size
        # (`args.window` characters on the left, the character in question, and
        # `args.window` characters on the right), where each character is
        # represented by a `torch.int64` index. To suitably represent the
        # characters, you can:
        # - Convert the character indices into _one-hot encoding_, which you can
        #   achieve by using `torch.nn.functional.one_hot` on the characters,
        #   and then concatenate the one-hot encodings of the window characters.
        # - Alternatively, you can experiment with `torch.nn.Embedding`s (an
        #   efficient implementation of one-hot encoding followed by a Dense layer)
        #   and flattening afterwards, or suitably using `torch.nn.EmbeddingBag`.
        
        # Character Embeddings
        embedding_dim = 16
        self.embedding = torch.nn.Embedding(num_embeddings=args.alphabet_size, embedding_dim=embedding_dim)
        
        input_size = embedding_dim * (2 * args.window + 1)
        self.layer1 = torch.nn.Linear(input_size, 64)
        self.layer2 = torch.nn.Linear(64, 1)
        self.flatten = torch.nn.Flatten()
        
    def forward(self, windows: torch.Tensor) -> torch.Tensor:
        # TODO: Implement the forward pass.
        x = self.embedding(windows)  # Convert character indices to embeddings
        x = self.flatten(x)
        x = torch.relu(self.layer1(x))
        x = self.layer2(x)
        return x

def train_model(model, optimizer, loss_fn, train_generator, epoch):
    model.train()
    total_loss = 0
    for inputs, targets in train_generator:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs.squeeze(-1), targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(train_generator)
    print(f"Epoch {epoch+1}: train_loss={avg_loss:.4f}")
    return avg_loss

def evaluate_model(model, dev_generator, loss_fn):
    model.eval()
    total_loss = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in dev_generator:
            outputs = model(inputs)
            loss = loss_fn(outputs.squeeze(-1), targets)
            total_loss += loss.item()
            total += targets.size(0)
    avg_loss = total_loss / len(dev_generator)
    print(f"dev_loss={avg_loss:.4f}")
    return avg_loss

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

    # Load the data. The default label dtype of torch.float32 is suitable for binary classification,
    # but you should change it to torch.int64 if you use 2-class classification (CrossEntropyLoss).
    uppercase_data = UppercaseData(args.window, args.alphabet_size, label_dtype=torch.float32)
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    # Instead of using
    #   train = torch.utils.data.DataLoader(
    #     torch.utils.data.StackDataset(uppercase_data.train.windows, uppercase_data.train.labels),
    #     batch_size=args.batch_size, shuffle=True)
    # we use the BatchGenerator, which is about an order of magnitude faster.
    train = BatchGenerator(uppercase_data.train.windows.to(device), uppercase_data.train.labels.to(device), args.batch_size, shuffle=True)
    dev = BatchGenerator(uppercase_data.dev.windows.to(device), uppercase_data.dev.labels.to(device), args.batch_size, shuffle=False)
    test = BatchGenerator(uppercase_data.test.windows.to(device), uppercase_data.test.labels.to(device), args.batch_size, shuffle=False)

    # TODO: Implement a suitable model, optionally including regularization, select
    # good hyperparameters, and train the model.
    
    model = Model(args).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    
    # Training loop with evaluation after each epoch
    for epoch in range(args.epochs):
        train_model(model, optimizer, loss_fn, train, epoch)
        evaluate_model(model, dev, loss_fn)

    # TODO: Generate correctly capitalized test set. Use `uppercase_data.test.text`
    # as input, capitalize suitable characters, and write the result to `predictions_file`
    # (which is by default `uppercase_test.txt` in the `args.logdir` directory).
    
    model.eval()
    all_preds = []
    with torch.no_grad():
        for inputs, _ in test:
            outputs = model(inputs)
            # Apply sigmoid, threshold, and squeeze to get integer predictions.
            preds = (torch.sigmoid(outputs) > 0.5).squeeze(-1).long()
            all_preds.append(preds)

    all_preds = torch.cat(all_preds, dim=0).cpu().numpy()

    # Convert the test text into a NumPy array of characters.
    text_chars = np.array(list(uppercase_data.test.text))

    # Use NumPy indexing to uppercase characters where the prediction is 1.
    text_chars[all_preds == 1] = np.char.upper(text_chars[all_preds == 1])

    # Join the characters back into a single string.
    final_text = ''.join(text_chars.tolist())
    
    os.makedirs(args.logdir, exist_ok=True)
    with open(os.path.join(args.logdir, "uppercase_test.txt"), "w", encoding="utf-8") as predictions_file:
        predictions_file.write(final_text)

if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    main(main_args)
    