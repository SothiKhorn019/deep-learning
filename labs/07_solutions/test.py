#!/usr/bin/env python3
import argparse
import datetime
import os
import re

import torch
import torch.nn as nn
import torch.optim as optim
import torchmetrics
from torch.nn.utils.rnn import pack_sequence, pad_packed_sequence, PackedSequence

import npfl138
npfl138.require_version("2425.7.2")
from npfl138.datasets.morpho_dataset import MorphoDataset
#from npfl138.datasets.morpho_analyzer import MorphoAnalyzer  # Optional: for future enhancements

# Set up argument parser.
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=10, type=int, help="Batch size.")
parser.add_argument("--epochs", default=5, type=int, help="Number of epochs.")
parser.add_argument("--max_sentences", default=None, type=int, help="Maximum number of sentences to load.")
parser.add_argument("--rnn", default="LSTM", choices=["LSTM", "GRU"], help="RNN layer type.")
parser.add_argument("--rnn_dim", default=64, type=int, help="RNN layer dimension.")
parser.add_argument("--we_dim", default=128, type=int, help="Word embedding dimension.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")

###############################################################################
# Model definition using PackedSequence.
###############################################################################
class Model(npfl138.TrainableModule):
    def __init__(self, args: argparse.Namespace, train: MorphoDataset.Dataset) -> None:
        super().__init__()
        # Save the vocabularies for prediction.
        self.word_vocab = train.words.string_vocab
        self.tag_vocab = train.tags.string_vocab

        # Create the word embedding layer.
        self._word_embedding = nn.Embedding(
            num_embeddings=len(self.word_vocab),
            embedding_dim=args.we_dim
        )

        # Choose the RNN type and create a bidirectional RNN.
        rnn_class = nn.LSTM if args.rnn == "LSTM" else nn.GRU
        self._word_rnn = rnn_class(
            input_size=args.we_dim,
            hidden_size=args.rnn_dim,
            bidirectional=True,
            batch_first=False  # PackedSequence expects batch_first=False.
        )

        # Create the output linear layer.
        # After summing the forward and backward outputs, the feature dimension is args.rnn_dim.
        self._output_layer = nn.Linear(
            in_features=args.rnn_dim,
            out_features=len(self.tag_vocab)
        )

    def forward(self, word_ids: PackedSequence) -> PackedSequence:
        # Embed the input word ids.
        embedded = word_ids._replace(data=self._word_embedding(word_ids.data))
        # Process the embeddings with the RNN.
        rnn_output, _ = self._word_rnn(embedded)
        # rnn_output.data is of shape [total_length, 2 * rnn_dim].
        # Sum the forward and backward outputs.
        hidden_size = self._word_rnn.hidden_size
        summed_data = rnn_output.data[:, :hidden_size] + rnn_output.data[:, hidden_size:]
        summed_output = rnn_output._replace(data=summed_data)
        # Pass through the output layer to get logits.
        logits = self._output_layer(summed_output.data)
        return summed_output._replace(data=logits)

    def compute_loss(self, y_pred, y_true, *xs):
        # y_pred and y_true are PackedSequence objects: use their data.
        return super().compute_loss(y_pred.data, y_true.data, *xs)

    def compute_metrics(self, y_pred, y_true, *xs):
        # Use raw data for metric computation.
        return super().compute_metrics(y_pred.data, y_true.data, *xs)

    def predict(self, dataset, data_with_labels=False):
        """
        Predict the POS tags for each sentence in `dataset`.
        Each sentence is processed individually, and the output for a sentence is a NumPy array
        with shape [num_tags, sentence_length], so that taking argmax over axis 0 yields the tag.
        """
        self.eval()
        predictions = []
        with torch.no_grad():
            for sentence in dataset.words.strings:
                # Convert each word into an index using the training word vocabulary.
                indices = [self.word_vocab.index(w) for w in sentence]
                word_tensor = torch.tensor(indices, dtype=torch.long)
                # Pack the single sentence.
                packed_input = pack_sequence([word_tensor], enforce_sorted=False)
                packed_output = self.forward(packed_input)
                # Unpack the output; pad_packed_sequence returns (tensor, lengths).
                output, lengths = pad_packed_sequence(packed_output, batch_first=True)
                # output shape: (1, L, num_tags). Remove the batch dimension.
                output = output.squeeze(0)  # shape becomes (L, num_tags)
                # Transpose to have shape (num_tags, L) as expected.
                output = output.transpose(0, 1)
                predictions.append(output.cpu().numpy())
        return predictions

###############################################################################
# Dataset preparation using PackedSequence.
###############################################################################
class TrainableDataset(npfl138.TransformedDataset):
    def transform(self, example):
        # Transform a single example into tensors of word and tag ids.
        word_ids = torch.tensor(
            [self.dataset.words.string_vocab.index(word) for word in example["words"]],
            dtype=torch.long
        )
        tag_ids = torch.tensor(
            [self.dataset.tags.string_vocab.index(tag) for tag in example["tags"]],
            dtype=torch.long
        )
        return word_ids, tag_ids

    def collate(self, batch):
        # Unpack the batch into separate lists of word and tag tensors.
        word_ids, tag_ids = zip(*batch)
        # Pack the sequences (handling varying lengths) using pack_sequence.
        packed_words = torch.nn.utils.rnn.pack_sequence(word_ids, enforce_sorted=False)
        packed_tags = torch.nn.utils.rnn.pack_sequence(tag_ids, enforce_sorted=False)
        return packed_words, packed_tags

###############################################################################
# Main training and competition prediction routine.
###############################################################################
def main(args: argparse.Namespace) -> dict[str, float]:
    # Set the random seed and number of threads.
    npfl138.startup(args.seed, args.threads)
    npfl138.global_keras_initializers()

    # Create a unique log directory.
    args.logdir = os.path.join("logs", "{}-{}-{}".format(
        os.path.basename(globals().get("__file__", "notebook")),
        datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", k), v) for k, v in sorted(vars(args).items())))
    ))
    os.makedirs(args.logdir, exist_ok=True)
    # device = torch.device("cpu")

    # Load the PDT dataset.
    morpho = MorphoDataset("czech_pdt", max_sentences=args.max_sentences)
    # analyses = MorphoAnalyzer("czech_pdt_analyses")  # Uncomment if you need analyzer outputs.

    # Prepare training and development datasets.
    train = TrainableDataset(morpho.train).dataloader(batch_size=args.batch_size, shuffle=True)
    dev = TrainableDataset(morpho.dev).dataloader(batch_size=args.batch_size)

    # Create the model.
    model = Model(args, morpho.train)
    
    # model.to(device)

    # Configure training: Adam optimizer, CrossEntropyLoss, and a multiclass Accuracy metric.
    model.configure(
        optimizer=optim.Adam(model.parameters()),
        loss=nn.CrossEntropyLoss(),
        metrics={"accuracy": torchmetrics.Accuracy(task="multiclass", num_classes=len(morpho.train.tags.string_vocab))},
        logdir=args.logdir,
    )

    # Train the model.
    logs = model.fit(train, dev=dev, epochs=args.epochs)

    # Generate predictions for the test set.
    predictions = model.predict(morpho.test, data_with_labels=True)
    predictions_filepath = os.path.join(args.logdir, "tagger_competition.txt")
    with open(predictions_filepath, "w", encoding="utf-8") as predictions_file:
        for predicted_tags, words in zip(predictions, morpho.test.words.strings):
            # predicted_tags has shape [num_tags, sentence_length]. Take argmax along axis 0 for each token.
            tag_indices = predicted_tags.argmax(axis=0)
            for tag_idx in tag_indices:
                # Convert tag index to string using the training tag vocabulary.
                tag_str = morpho.train.tags.string_vocab.string(tag_idx)
                print(tag_str, file=predictions_file)
            print(file=predictions_file)

    return {metric: value for metric, value in logs.items() if metric.startswith("dev_")}


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    main(main_args)
