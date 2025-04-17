#!/usr/bin/env python3
# Group IDs: 31ff17c9-b0b8-449e-b0ef-8a1aa1e14eb3, 5b78caaa-8040-46f7-bf54-c13e183bbbf8
import argparse
import datetime
import os
import re

import torch
import torchmetrics
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.utils.rnn
from torch.utils.data import DataLoader

import npfl138
npfl138.require_version("2425.7.2")
from npfl138.datasets.morpho_dataset import MorphoDataset

# TODO: Define reasonable defaults and optionally more parameters.
# Also, you can set the number of threads to 0 to use all your CPU cores.
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=10, type=int, help="Batch size.")
parser.add_argument("--epochs", default=5, type=int, help="Number of epochs.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
# added arument
parser.add_argument("--cle_dim", default=32, type=int, help="CLE embedding dimension.")
parser.add_argument("--max_sentences", default=None, type=int, help="Maximum number of sentences to load.")
parser.add_argument("--rnn", default="LSTM", choices=["LSTM", "GRU"], help="RNN layer type.")
parser.add_argument("--rnn_dim", default=64, type=int, help="RNN layer dimension.")
parser.add_argument("--we_dim", default=64, type=int, help="Word embedding dimension.")
parser.add_argument("--word_masking", default=0.0, type=float, help="Mask words with the given probability.")

class Model(npfl138.TrainableModule):
    class MaskElements(torch.nn.Module):
        """A layer randomly masking elements with a given value."""
        def __init__(self, mask_probability, mask_value):
            super().__init__()
            self._mask_probability = mask_probability
            self._mask_value = mask_value

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            # Only mask during training and when mask_probability > 0.
            if self.training and self._mask_probability:
                # Generate a mask of floats in [0,1] of the same shape as inputs.
                mask = torch.rand_like(inputs, dtype=torch.float32)
                # Replace elements whose random value is less than mask_probability.
                inputs = torch.where(
                    mask < self._mask_probability,
                    torch.tensor(self._mask_value, dtype=inputs.dtype, device=inputs.device),
                    inputs,
                )
            return inputs

    def __init__(self, args: argparse.Namespace, train: MorphoDataset.Dataset) -> None:
        super().__init__()

        # Save padding indices (obtained from the vocabularies in the training dataset).
        self._char_pad = train.words.char_vocab.PAD
        self._word_pad = train.words.string_vocab.PAD

        # Create word masking layer.
        self._word_masking = Model.MaskElements(args.word_masking, MorphoDataset.UNK)

        # Create a character embedding layer for CLE.
        self._char_embedding = torch.nn.Embedding(
            num_embeddings=len(train.words.char_vocab),
            embedding_dim=args.cle_dim,
            padding_idx=self._char_pad
        )

        # Create a bidirectional GRU for character-level encoding.
        self._char_rnn = torch.nn.GRU(
            input_size=args.cle_dim,
            hidden_size=args.cle_dim,
            batch_first=True,
            bidirectional=True
        )

        # Create a word embedding layer.
        self._word_embedding = torch.nn.Embedding(
            num_embeddings=len(train.words.string_vocab),
            embedding_dim=args.we_dim,
            padding_idx=self._word_pad
        )

        # Create a word-level RNN. Choose LSTM or GRU based on args.rnn.
        input_size = args.we_dim + 2 * args.cle_dim  # concatenation of word and CLE embeddings.
        if args.rnn == "LSTM":
            self._word_rnn = torch.nn.LSTM(
                input_size=input_size,
                hidden_size=args.rnn_dim,
                batch_first=True,
                bidirectional=True
            )
        else:  # GRU option.
            self._word_rnn = torch.nn.GRU(
                input_size=input_size,
                hidden_size=args.rnn_dim,
                batch_first=True,
                bidirectional=True
            )

        # Create the output linear layer for tag prediction.
        # The RNN output will be summed from forward and backward directions.
        self._output_layer = torch.nn.Linear(args.rnn_dim, len(train.tags.string_vocab))

    def forward(self, word_ids: torch.Tensor, unique_words: torch.Tensor, word_indices: torch.Tensor) -> torch.Tensor:
        # Apply word masking.
        masked_word_ids = self._word_masking(word_ids)

        # Embed the (possibly masked) word ids.
        word_emb = self._word_embedding(masked_word_ids)  # shape: [batch, seq_len, we_dim]

        # Embed unique words (for character-level encoding).
        char_emb = self._char_embedding(unique_words)  # shape: [num_unique, max_word_length, cle_dim]

        # Compute lengths of each unique word (by counting non-pad positions).
        char_lengths = (unique_words != self._char_pad).sum(dim=1)
        packed_chars = torch.nn.utils.rnn.pack_padded_sequence(
            char_emb, char_lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        # Process with the character-level RNN.
        _, h_n = self._char_rnn(packed_chars)
        # h_n has shape [num_directions, num_unique, cle_dim]; concatenate forward & backward.
        char_repr = torch.cat([h_n[0], h_n[1]], dim=1)  # shape: [num_unique, 2 * cle_dim]

        # Build CLE for each word in the sentence by indexing the unique words representation.
        cle = F.embedding(word_indices, char_repr)  # shape: [batch, seq_len, 2 * cle_dim]

        # Concatenate word embeddings with CLE.
        concat_emb = torch.cat([word_emb, cle], dim=2)  # shape: [batch, seq_len, we_dim + 2*cle_dim]

        # Compute sentence lengths (by counting non-pad tokens).
        sentence_lengths = (word_ids != self._word_pad).sum(dim=1)
        packed_sent = torch.nn.utils.rnn.pack_padded_sequence(
            concat_emb, sentence_lengths.cpu(), batch_first=True, enforce_sorted=False
        )

        # Process with the word-level RNN.
        rnn_out, _ = self._word_rnn(packed_sent)
        # Unpack the RNN outputs.
        rnn_out, _ = torch.nn.utils.rnn.pad_packed_sequence(rnn_out, batch_first=True)
        # Sum forward and backward outputs.
        rnn_sum = rnn_out[:, :, :self._word_rnn.hidden_size] + rnn_out[:, :, self._word_rnn.hidden_size:]

        # Produce logits for tag prediction.
        logits = self._output_layer(rnn_sum)  # shape: [batch, seq_len, num_tags]
        # Rearrange dimensions to [batch, num_tags, seq_len] as required by the loss.
        logits = logits.transpose(1, 2)
        return logits

class TrainableDataset(npfl138.TransformedDataset):
    def transform(self, example):
        # Transform an example (a dictionary with keys "words", "lemmas", "tags")
        # into a tuple: (word_ids, words, tag_ids)
        word_ids = torch.tensor(
            [self.dataset.words.string_vocab.index(token) for token in example["words"]],
            dtype=torch.long
        )
        tag_ids = torch.tensor(
            [self.dataset.tags.string_vocab.index(tag) for tag in example["tags"]],
            dtype=torch.long
        )
        # We return the original word strings as well (for CLE construction).
        return word_ids, example["words"], tag_ids

    def collate(self, batch):
        # Batch is a list of tuples: (word_ids, words, tag_ids)
        word_ids, words, tag_ids = zip(*batch)
        # Pad word_ids.
        word_ids = torch.nn.utils.rnn.pad_sequence(
            word_ids, batch_first=True, padding_value=self.dataset.words.string_vocab.PAD
        )
        # Create CLE inputs using the provided cle_batch function.
        unique_words, words_indices = self.dataset.cle_batch(words)
        # Pad tag_ids.
        tag_ids = torch.nn.utils.rnn.pad_sequence(
            tag_ids, batch_first=True, padding_value=self.dataset.tags.string_vocab.PAD
        )
        return (word_ids, unique_words, words_indices), tag_ids

def predict_test(model: Model, test_dataset: MorphoDataset.Dataset, device: torch.device) -> list[list[int]]:
    """
    For each test sentence, compute model predictions.
    Returns a list of lists of predicted tag indices.
    """
    model.eval()
    predictions = []
    # Loop over test sentences (raw list of tokens).
    for sentence in test_dataset.words.strings:
        # Convert sentence tokens to indices using the training vocabulary.
        word_ids = torch.tensor(
            [test_dataset.words.string_vocab.index(token) for token in sentence],
            dtype=torch.long, device=device
        ).unsqueeze(0)
        # Prepare CLE inputs. The cle_batch function expects a list of sentences.
        unique_words, word_indices = test_dataset.cle_batch([sentence])
        unique_words = unique_words.to(device)
        word_indices = word_indices.to(device)
        with torch.no_grad():
            # Forward pass.
            # Output shape: [1, num_tags, seq_len]
            output = model(word_ids, unique_words, word_indices)
        # Predict tags: take argmax over the num_tags dimension.
        pred_tags = output.argmax(dim=1).squeeze(0).tolist()  # list of predicted indices
        predictions.append(pred_tags)
    return predictions

def main(args: argparse.Namespace) -> dict[str, float]:
    # Set random seed and number of threads.
    npfl138.startup(args.seed, args.threads)
    npfl138.global_keras_initializers()

    # Create a unique log directory.
    args.logdir = os.path.join("logs", "{}-{}-{}".format(
        os.path.basename(globals().get("__file__", "notebook")),
        datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", k), v) for k, v in sorted(vars(args).items())))
    ))

    # Load the Czech PDT data.
    morpho = MorphoDataset("czech_pdt", max_sentences=args.max_sentences)

    # Prepare training and development data using the TrainableDataset wrapper.
    train_loader = TrainableDataset(morpho.train).dataloader(batch_size=args.batch_size, shuffle=True)
    dev_loader   = TrainableDataset(morpho.dev).dataloader(batch_size=args.batch_size)

    # TODO: Create the model and train it.
    model = Model(args, morpho.train)

    # Configure training: optimizer, loss, and metrics.
    model.configure(
        optimizer=optim.Adam(model.parameters()),
        loss=torch.nn.CrossEntropyLoss(ignore_index=morpho.PAD),
        metrics={"accuracy": torchmetrics.Accuracy(
            task="multiclass",
            num_classes=len(morpho.train.tags.string_vocab),
            ignore_index=morpho.train.tags.string_vocab.PAD
        )},
        logdir=args.logdir,
    )

    # Train the model.
    logs = model.fit(train_loader, dev=dev_loader, epochs=args.epochs)

    # After training, generate test set predictions.
    device = next(model.parameters()).device
    test_predictions = predict_test(model, morpho.test, device)

    # Generate test set annotations, but in `args.logdir` to allow parallel execution.
    os.makedirs(args.logdir, exist_ok=True)
    pred_path = os.path.join(args.logdir, "tagger_competition.txt")
    with open(pred_path, "w", encoding="utf-8") as f:
        # TODO: Predict the tags on the test set. The following code assumes you use the same
        # output structure as in `tagger_we`, i.e., that for each sentence, the predictions are
        # a Numpy vector of shape `[num_tags, sentence_len_or_more]`, where `sentence_len_or_more`
        # is the length of the corresponding batch. (FYI, if you instead used the `packed` variant,
        # the prediction for each sentence is a vector of shape `[exactly_sentence_len, num_tags]`.)
        for pred_tags, words in zip(test_predictions, morpho.test.words.strings):
            for tag in pred_tags[:len(words)]:
                f.write(morpho.train.tags.string_vocab.string(tag) + "\n")
            f.write("\n")

if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    main(main_args)
