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

os.environ.pop("PYTORCH_MPS_HIGH_WATERMARK_RATIO", None)

parser = argparse.ArgumentParser()
# These arguments are set by ReCodEx or via command line.
parser.add_argument("--batch_size", default=10, type=int, help="Batch size.")
parser.add_argument("--cle_dim", default=32, type=int, help="CLE embedding dimension.")
parser.add_argument("--epochs", default=1, type=int, help="Number of epochs.")
parser.add_argument("--max_sentences", default=None, type=int, help="Maximum number of sentences to load.")
parser.add_argument("--recodex", default=False, action="store_true", help="Evaluation in ReCodEx.")
parser.add_argument("--rnn", default="LSTM", choices=["LSTM", "GRU"], help="RNN layer type.")
parser.add_argument("--rnn_dim", default=64, type=int, help="RNN layer dimension.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
parser.add_argument("--we_dim", default=64, type=int, help="Word embedding dimension.")
parser.add_argument("--word_masking", default=0.0, type=float, help="Mask words with the given probability.")


###############################################################################
# Model definition following tagger_cle.packed.py structure.
###############################################################################
class Model(npfl138.TrainableModule):
    class MaskElements(nn.Module):
        """A layer for randomly masking tensor elements with a given probability."""
        def __init__(self, mask_probability, mask_value):
            super().__init__()
            self._mask_probability = mask_probability
            self._mask_value = mask_value

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            if self.training and self._mask_probability:
                mask = torch.rand_like(inputs, dtype=torch.float32) < self._mask_probability
                inputs = torch.where(mask, self._mask_value, inputs)
            return inputs

    def __init__(self, args: argparse.Namespace, train: MorphoDataset.Dataset) -> None:
        super().__init__()
        # Create word masking layer; using MorphoDataset.UNK as the mask value.
        self._word_masking = self.MaskElements(args.word_masking, MorphoDataset.UNK)
        # Character-level embedding for CLE.
        self._char_embedding = nn.Embedding(len(train.words.char_vocab), args.cle_dim)
        # Bidirectional GRU for character-level processing.
        self._char_rnn = nn.GRU(args.cle_dim, args.cle_dim, bidirectional=True, batch_first=False)
        # Word-level embedding.
        self._word_embedding = nn.Embedding(len(train.words.string_vocab), args.we_dim)
        # Word-level RNN: its input is the concatenation of word embeddings with CLE features.
        rnn_class = nn.LSTM if args.rnn == "LSTM" else nn.GRU
        self._word_rnn = rnn_class(args.we_dim + 2 * args.cle_dim, args.rnn_dim,
                                   bidirectional=True, batch_first=False)
        # Output linear layer to produce logits for each tag.
        self._output_layer = nn.Linear(args.rnn_dim, len(train.tags.string_vocab))

    def forward(self, 
                word_ids: PackedSequence,
                unique_words: PackedSequence,
                word_indices: PackedSequence) -> PackedSequence:
        # Mask input word IDs.
        masked_data = self._word_masking(word_ids.data)
        masked_word_ids = word_ids._replace(data=masked_data)
        # Get word-level embeddings.
        word_embeds = self._word_embedding(masked_word_ids.data)
        # Process character-level embeddings.
        char_embeds = self._char_embedding(unique_words.data)
        char_embeds_packed = unique_words._replace(data=char_embeds)
        _, char_hidden = self._char_rnn(char_embeds_packed)
        # Concatenate hidden states from forward and backward directions.
        cle = torch.cat([char_hidden[0], char_hidden[1]], dim=1)
        # For every word in the sentence, gather its corresponding CLE representation.
        cle_for_words = cle[word_indices.data]
        # Concatenate word embeddings with CLE features.
        combined = torch.cat([word_embeds, cle_for_words], dim=1)
        combined_packed = masked_word_ids._replace(data=combined)
        # Process the concatenated embeddings with the word-level RNN.
        rnn_output, _ = self._word_rnn(combined_packed)
        hidden_size = self._word_rnn.hidden_size
        # Sum forward and backward outputs.
        summed_data = rnn_output.data[:, :hidden_size] + rnn_output.data[:, hidden_size:]
        summed_output = rnn_output._replace(data=summed_data)
        # Pass through the output linear layer.
        logits = self._output_layer(summed_output.data)
        return summed_output._replace(data=logits)

    def compute_loss(self, y_pred, y_true, *xs):
        return super().compute_loss(y_pred.data, y_true.data, *xs)

    def compute_metrics(self, y_pred, y_true, *xs):
        return super().compute_metrics(y_pred.data, y_true.data, *xs)

    def predict(self, dataset, data_with_labels=False):
        """
        Predict POS tags for each sentence in the dataset.
        Returns a list of NumPy arrays, one per sentence with shape [num_tags, sentence_length].
        """
        self.eval()
        predictions = []
        # Obtain model's device (now should be MPS).
        device = next(self.parameters()).device
        word_vocab = dataset.words.string_vocab
        with torch.no_grad():
            for sentence in dataset.words.strings:
                # Create a tensor of word IDs on the current device.
                indices = [word_vocab.index(w) for w in sentence]
                word_tensor = torch.tensor(indices, dtype=torch.long, device=device)
                # Pack the sentence.
                word_ids = pack_sequence([word_tensor], enforce_sorted=False)
                # Get CLE inputs using the dataset's built-in CLE batch function.
                unique_words, words_indices = dataset.cle_batch_packed([sentence])
                unique_words = unique_words._replace(data=unique_words.data.to(device))
                words_indices = words_indices._replace(data=words_indices.data.to(device))
                # Forward pass.
                output_packed = self.forward(word_ids, unique_words, words_indices)
                output, _ = pad_packed_sequence(output_packed, batch_first=True)
                output = output.squeeze(0)  # shape: (L, num_tags)
                # Transpose so rows correspond to tag classes.
                predictions.append(output.transpose(0, 1).cpu().numpy())
        return predictions


###############################################################################
# Dataset preparation following tagger_cle.packed.py
###############################################################################
class TrainableDataset(npfl138.TransformedDataset):
    def transform(self, example):
        # For each example: tensor of word IDs, original word list, tensor of tag IDs.
        word_ids = torch.tensor(
            [self.dataset.words.string_vocab.indices([word])[0] for word in example["words"]],
            dtype=torch.long)
        tag_ids = torch.tensor(
            [self.dataset.tags.string_vocab.indices([tag])[0] for tag in example["tags"]],
            dtype=torch.long)
        return word_ids, example["words"], tag_ids

    def collate(self, batch):
        word_ids, words, tag_ids = zip(*batch)
        word_ids = pack_sequence(word_ids, enforce_sorted=False)
        # Create CLE inputs using the dataset's CLE function.
        unique_words, words_indices = self.dataset.cle_batch_packed(words)
        tag_ids = pack_sequence(tag_ids, enforce_sorted=False)
        return (word_ids, unique_words, words_indices), tag_ids


###############################################################################
# Main training routine.
###############################################################################
def main(args: argparse.Namespace) -> dict[str, float]:
    npfl138.startup(args.seed, args.threads)
    npfl138.global_keras_initializers()

    args.logdir = os.path.join("logs", "{}-{}-{}".format(
        os.path.basename(globals().get("__file__", "notebook")),
        datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", k), v)
                  for k, v in sorted(vars(args).items())))
    ))
    os.makedirs(args.logdir, exist_ok=True)

    # Load the dataset using the appropriate dataset identifier (e.g. "czech_cac" for CLE).
    morpho = MorphoDataset("czech_cac", max_sentences=args.max_sentences)

    train = TrainableDataset(morpho.train).dataloader(batch_size=args.batch_size, shuffle=True)
    dev   = TrainableDataset(morpho.dev).dataloader(batch_size=args.batch_size)

    model = Model(args, morpho.train)
    # Force the model to use the MPS device.
    device = torch.device("mps")
    model.to(device)

    model.configure(
        optimizer=optim.Adam(model.parameters()),
        loss=nn.CrossEntropyLoss(),
        metrics={"accuracy": torchmetrics.Accuracy(
            task="multiclass", num_classes=len(morpho.train.tags.string_vocab))},
        logdir=args.logdir,
    )

    logs = model.fit(train, dev=dev, epochs=args.epochs)

    # Generate predictions for the test set.
    predictions = model.predict(morpho.test, data_with_labels=True)
    predictions_filepath = os.path.join(args.logdir, "tagger_competition.txt")
    with open(predictions_filepath, "w", encoding="utf-8") as predictions_file:
        for predicted_tags, words in zip(predictions, morpho.test.words.strings):
            tag_indices = predicted_tags[:, :len(words)].argmax(axis=0)
            for tag_idx in tag_indices:
                tag_str = morpho.train.tags.string_vocab.string(tag_idx)
                print(tag_str, file=predictions_file)
            print(file=predictions_file)

    return {metric: value for metric, value in logs.items() if metric.startswith("dev_")}


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    main(main_args)
