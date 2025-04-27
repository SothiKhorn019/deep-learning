#!/usr/bin/env python3
import argparse
import datetime
import os
import re
import math

import torch
import torchmetrics
import torch.nn.functional as F

import npfl138

npfl138.require_version("2425.9")
from npfl138.datasets.morpho_dataset import MorphoDataset

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--batch_size", default=10, type=int, help="Batch size.")
parser.add_argument("--cle_dim", default=64, type=int, help="CLE embedding dimension.")
parser.add_argument("--epochs", default=10, type=int, help="Number of epochs.")
parser.add_argument(
    "--max_sentences",
    default=None,
    type=int,
    help="Maximum number of sentences to load.",
)
parser.add_argument(
    "--recodex", default=False, action="store_true", help="Evaluation in ReCodEx."
)
parser.add_argument("--rnn_dim", default=64, type=int, help="RNN layer dimension.")
parser.add_argument("--seed", default=41, type=int, help="Random seed.")
parser.add_argument(
    "--show_results_every_batch",
    default=10,
    type=int,
    help="Show results every given batch.",
)
parser.add_argument(
    "--tie_embeddings",
    default=False,
    action="store_true",
    help="Tie target embeddings.",
)
parser.add_argument(
    "--threads", default=1, type=int, help="Maximum number of threads to use."
)
# If you add more arguments, ReCodEx will keep them with your default values.


class WithAttention(torch.nn.Module):
    """A class adding Bahdanau attention to a given RNN cell."""

    def __init__(self, cell, attention_dim):
        super().__init__()
        self._cell = cell

        # TODO: Define
        # - `self._project_encoder_layer` as a linear layer with `cell.hidden_size` inputs
        #   and `attention_dim` outputs.
        # - `self._project_decoder_layer` as a linear layer with `cell.hidden_size` inputs
        #   and `attention_dim` outputs
        # - `self._output_layer` as a linear layer with `attention_dim` inputs and 1 output

        self._project_encoder_layer = torch.nn.Linear(cell.hidden_size, attention_dim)
        self._project_decoder_layer = torch.nn.Linear(cell.hidden_size, attention_dim)
        self._output_layer = torch.nn.Linear(attention_dim, 1)

    def setup_memory(self, encoded):
        self._encoded = encoded
        # TODO: Pass the `encoded` through the `self._project_encoder_layer` and store
        # the result as `self._encoded_projected`.
        self._encoded_projected = self._project_encoder_layer(encoded)

    def forward(self, inputs, states):
        # TODO: Compute the attention.
        # - According to the definition, we need to project the encoder states, but we have
        #   already done that in `setup_memory`, so we just take `self._encoded_projected`.
        # - Compute projected decoder state by passing the given state through the `self._project_decoder_layer`.
        # - Sum the two projections. However, you have to deal with the fact that the first projection has
        #   shape `[batch_size, input_sequence_len, attention_dim]`, while the second projection has
        #   shape `[batch_size, attention_dim]`. The best solution is capable of creating the sum
        #   directly without creating any intermediate tensor.
        # - Pass the sum through the `torch.tanh` and then through the `self._output_layer`.
        # - Then, run softmax activation, generating `weights`.
        # - Multiply the original (non-projected) encoder states `self._encoded` with `weights` and sum
        #   the result in the axis corresponding to characters, generating `attention`. Therefore,
        #   `attention` is a fixed-size representation for every batch element, independently on
        #   how many characters the corresponding input word had.
        # - Finally, concatenate `inputs` and `attention` (in this order), and call the `self._cell`
        #   on this concatenated input and the `states`, returning the result.

        # inputs: [batch, inp_dim]; states: [batch, hidden]
        # project decoder state: [batch, attn_dim]
        dec_proj = self._project_decoder_layer(states)
        # expand and sum: [batch, src_len, attn_dim]
        energy = torch.tanh(self._encoded_projected + dec_proj.unsqueeze(1))
        # score: [batch, src_len, 1]
        scores = self._output_layer(energy)
        # weights: [batch, src_len, 1]
        weights = torch.softmax(scores, dim=1)
        # context: weighted sum of original encoded
        # encoded: [batch, src_len, hidden]
        attention = torch.sum(weights * self._encoded, dim=1)
        # concat input and context
        cell_input = torch.cat([inputs, attention], dim=1)
        # call underlying cell
        new_state = self._cell(cell_input, states)
        return new_state


class Model(npfl138.TrainableModule):
    def __init__(self, args: argparse.Namespace, train: MorphoDataset.Dataset) -> None:
        super().__init__()
        self._source_vocab = train.words.char_vocab
        self._target_vocab = train.lemmas.char_vocab

        # TODO(lemmatizer_noattn): Define
        # - `self._source_embedding` as an embedding layer of source characters into `args.cle_dim` dimensions
        # - `self._source_rnn` as a bidirectional GRU with `args.rnn_dim` units processing embedded source chars
        self._source_embedding = torch.nn.Embedding(
            num_embeddings=len(self._source_vocab), embedding_dim=args.cle_dim
        )
        self._source_rnn = torch.nn.GRU(
            input_size=args.cle_dim,
            hidden_size=args.rnn_dim,
            bidirectional=True,
            batch_first=True,
        )

        # TODO: Define
        # - `self._target_rnn_cell` as a `WithAttention` with `attention_dim=args.rnn_dim`, employing as the
        #   underlying cell the `torch.nn.GRUCell` with `args.rnn_dim`. The cell will process concatenated
        #   target character embeddings and the result of the attention mechanism.
        base_cell = torch.nn.GRUCell(args.cle_dim + args.rnn_dim, args.rnn_dim)
        self._target_rnn_cell = WithAttention(base_cell, attention_dim=args.rnn_dim)

        # TODO(lemmatizer_noattn): Then define
        # - `self._target_output_layer` as a linear layer into as many outputs as there are unique target chars
        self._target_output_layer = torch.nn.Linear(
            in_features=args.rnn_dim, out_features=len(self._target_vocab)
        )

        if not args.tie_embeddings:
            # TODO(lemmatizer_noattn): Define the `self._target_embedding` as an embedding layer of the target
            # characters into `args.cle_dim` dimensions.
            self._target_embedding = torch.nn.Embedding(
                num_embeddings=len(self._target_vocab), embedding_dim=args.cle_dim
            )
        else:
            assert (
                args.cle_dim == args.rnn_dim
            ), "When tying embeddings, cle_dim and rnn_dim must match."
            # TODO(lemmatizer_noattn): Create a function `self._target_embedding` computing the embedding of given
            # target characters. When called, use `torch.nn.functional.embedding` to suitably
            # index the shared embedding matrix `self._target_output_layer.weight`
            # multiplied by the square root of `args.rnn_dim`.
            scale = math.sqrt(args.rnn_dim)
            self._target_embedding = lambda x: F.embedding(
                x, self._target_output_layer.weight * scale
            )

        self._show_results_every_batch = args.show_results_every_batch
        self._batches = 0

    def forward(
        self, words: torch.Tensor, targets: torch.Tensor | None = None
    ) -> torch.Tensor:
        encoded = self.encoder(words)
        if targets is not None:
            return self.decoder_training(encoded, targets)
        else:
            return self.decoder_prediction(encoded, max_length=words.shape[1] + 10)

    def encoder(self, words: torch.Tensor) -> torch.Tensor:
        # TODO(lemmatizer_noattn): Embed the inputs using `self._source_embedding`.

        # TODO: Run the `self._source_rnn` on the embedded sequences, correctly handling
        # padding. Newly, the result should be encoding of every sequence element,
        # summing results in the opposite directions.

        # words: [batch, src_len]
        lengths = (words != MorphoDataset.PAD).sum(dim=1)
        embedded = self._source_embedding(words)
        packed = torch.nn.utils.rnn.pack_padded_sequence(
            embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_out, _ = self._source_rnn(packed)
        # unpack all outputs: [batch, src_len, 2*rnn_dim]
        outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(
            packed_out, batch_first=True, total_length=words.size(1)
        )
        # sum forward and backward
        fwd = outputs[:, :, : self._source_rnn.hidden_size]
        bwd = outputs[:, :, self._source_rnn.hidden_size :]
        return fwd + bwd

    def decoder_training(
        self, encoded: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        # TODO(lemmatizer_noattn): Generate inputs for the decoder, which are obtained from `targets` by
        # - prepending `MorphoDataset.BOW` as the first element of every batch example,
        # - dropping the last element of `targets`.

        # TODO: Pre-compute the projected encoder states in the attention by calling
        # the `setup_memory` of the `self._target_rnn_cell` on the `encoded` input.

        # TODO: Process the generated inputs by
        # - the `self._target_embedding` layer to obtain embeddings,
        # - repeatedly call the `self._target_rnn_cell` on the sequence of embedded
        #   inputs and the previous states, starting with state `encoded[:, 0]`,
        #   obtaining outputs for all target hidden states,
        # - the `self._target_output_layer` to obtain logits,
        # - finally, permute dimensions so that the logits are in the dimension 1,
        # and return the result.

        # prepare decoder inputs: prepend BOW, drop last
        batch_size, tgt_len = targets.size()
        bows = torch.full(
            (batch_size, 1), MorphoDataset.BOW, dtype=torch.long, device=targets.device
        )
        shifted = targets[:, :-1]
        decoder_inputs = torch.cat([bows, shifted], dim=1)
        # embed
        embedded = self._target_embedding(decoder_inputs)
        # setup attention memory
        self._target_rnn_cell.setup_memory(encoded)
        # initial state from encoder: first timestep
        state = encoded[:, 0]
        # collect decoder hidden states
        outputs = []
        for t in range(decoder_inputs.size(1)):
            input_t = embedded[:, t, :]
            state = self._target_rnn_cell(input_t, state)
            outputs.append(state)
        # stack: [batch, tgt_len, rnn_dim]
        hidden_seq = torch.stack(outputs, dim=1)
        # project to vocabulary
        logits = self._target_output_layer(hidden_seq)
        # [batch, tgt_len, vocab] -> [batch, vocab, tgt_len]
        return logits.permute(0, 2, 1)

    def decoder_prediction(
        self, encoded: torch.Tensor, max_length: int
    ) -> torch.Tensor:
        batch_size, src_len, _ = encoded.size()
        # setup attention memory
        self._target_rnn_cell.setup_memory(encoded)
        # initialize
        index = 0
        inputs = torch.full(
            (batch_size,), MorphoDataset.BOW, dtype=torch.long, device=encoded.device
        )
        states = encoded[:, 0]
        results = []
        result_lengths = torch.full(
            (batch_size,), max_length, dtype=torch.long, device=encoded.device
        )

        while index < max_length and torch.any(result_lengths == max_length):
            embedded = self._target_embedding(inputs)
            hidden = self._target_rnn_cell(embedded, states)
            states = hidden
            logits = self._target_output_layer(hidden)
            predictions = logits.argmax(dim=-1)
            results.append(predictions)
            # update lengths on first EOW
            result_lengths[
                (predictions == MorphoDataset.EOW) & (result_lengths > index)
            ] = (index + 1)
            # next step
            inputs = predictions
            index += 1

        results = torch.stack(results, dim=1)
        return results

    def compute_metrics(self, y_pred, y, *xs):
        if (
            self.training
        ):  # In training regime, convert logits to most likely predictions.
            y_pred = y_pred.argmax(dim=-2)
        # Compare the lemmas with the predictions using exact match accuracy.
        y_pred = y_pred[:, : y.shape[-1]]
        y_pred = torch.nn.functional.pad(
            y_pred, (0, y.shape[-1] - y_pred.shape[-1]), value=MorphoDataset.PAD
        )
        self.metrics["accuracy"].update(
            torch.all((y_pred == y) | (y == MorphoDataset.PAD), dim=-1)
        )
        return {
            name: metric.compute() for name, metric in self.metrics.items()
        }  # Return all metrics.

    def train_step(self, xs, y):
        result = super().train_step(xs, y)

        self._batches += 1
        if (
            self._show_results_every_batch
            and self._batches % self._show_results_every_batch == 0
        ):
            self.log_console(
                "{}: {} -> {}".format(
                    self._batches,
                    "".join(
                        self._source_vocab.strings(
                            xs[0][0][xs[0][0] != MorphoDataset.PAD].numpy(force=True)
                        )
                    ),
                    "".join(
                        self._target_vocab.strings(self.predict_step((xs[0][:1],))[0])
                    ),
                )
            )

        return result

    def test_step(self, xs, y):
        with torch.no_grad():
            y_pred = self.forward(*xs)
            return self.compute_metrics(y_pred, y, *xs)

    def predict_step(self, xs, as_numpy=True):
        with torch.no_grad():
            batch = self.forward(*xs)
            # Trim the predictions at the first EOW
            batch = [
                lemma[(lemma == MorphoDataset.EOW).cumsum(-1) == 0] for lemma in batch
            ]
            return [lemma.numpy(force=True) for lemma in batch] if as_numpy else batch


class TrainableDataset(npfl138.TransformedDataset):
    def __init__(self, dataset: MorphoDataset.Dataset, training: bool) -> None:
        super().__init__(dataset)
        self._training = training

    def transform(self, example):
        # TODO(lemmatizer_noattn): Return `example["words"]` as inputs and `example["lemmas"]` as targets.
        return example["words"], example["lemmas"]

    def collate(self, batch):
        # Construct a single batch, where `batch` is a list of examples generated by `transform`.
        # TODO(lemmatizer_noattn): The `words` are a list of list of strings. Flatten it into a single list of strings
        # and then map the characters to their indices using the `self.dataset.words.char_vocab` vocabulary.
        # Then create a tensor by padding the words to the length of the longest one in the batch.

        # TODO(lemmatizer_noattn): Process `lemmas` analogously to `words`, but use `self.dataset.lemmas.char_vocab`,
        # and additionally, append `MorphoDataset.EOW` to the end of each lemma.

        # TODO(lemmatizer_noattn): Return a pair (inputs, targets), where
        # - the inputs are words during inference and (words, lemmas) pair during training;
        # - the targets are lemmas.

        words_list, lemmas_list = zip(*batch)
        # Flatten and index
        words_flat = [w for sent in words_list for w in sent]
        lemmas_flat = [l for sent in lemmas_list for l in sent]
        src_vocab = self.dataset.words.char_vocab
        tgt_vocab = self.dataset.lemmas.char_vocab
        words_idx = [src_vocab.indices(list(w)) for w in words_flat]
        max_w = max(len(w) for w in words_idx)
        words_tensor = torch.tensor(
            [w + [MorphoDataset.PAD] * (max_w - len(w)) for w in words_idx],
            dtype=torch.long,
        )
        lemmas_idx = [
            tgt_vocab.indices(list(l)) + [MorphoDataset.EOW] for l in lemmas_flat
        ]
        max_l = max(len(l) for l in lemmas_idx)
        lemmas_tensor = torch.tensor(
            [l + [MorphoDataset.PAD] * (max_l - len(l)) for l in lemmas_idx],
            dtype=torch.long,
        )
        if self._training:
            inputs = (words_tensor, lemmas_tensor)
        else:
            inputs = (words_tensor,)
        targets = lemmas_tensor
        return inputs, targets


def main(args: argparse.Namespace) -> dict[str, float]:
    # Set the random seed and the number of threads.
    npfl138.startup(args.seed, args.threads)
    npfl138.global_keras_initializers()

    # Create logdir name.
    args.logdir = os.path.join(
        "logs",
        "{}-{}-{}".format(
            os.path.basename(globals().get("__file__", "notebook")),
            datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
            ",".join(
                (
                    "{}={}".format(re.sub("(.)[^_]*_?", r"\1", k), v)
                    for k, v in sorted(vars(args).items())
                )
            ),
        ),
    )

    # Load the data.
    morpho = MorphoDataset("czech_cac", max_sentences=args.max_sentences)

    # Prepare the data for training.
    train = TrainableDataset(morpho.train, training=True).dataloader(
        batch_size=args.batch_size, shuffle=True
    )
    dev = TrainableDataset(morpho.dev, training=False).dataloader(
        batch_size=args.batch_size
    )

    # Create the model and train.
    model = Model(args, morpho.train)

    model.configure(
        # TODO(lemmatizer_noattn): Create the Adam optimizer.
        optimizer=torch.optim.Adam(model.parameters()),
        # TODO(lemmatizer_noattn): Use the usual `torch.nn.CrossEntropyLoss` loss function. Additionally,
        # pass `ignore_index=morpho.PAD` to the constructor so that the padded
        # tags are ignored during the loss computation.
        loss=torch.nn.CrossEntropyLoss(ignore_index=MorphoDataset.PAD),
        # TODO(lemmatizer_noattn): Create a `torchmetrics.MeanMetric()` metric, where we will manually
        # collect lemmatization accuracy.
        metrics={"accuracy": torchmetrics.MeanMetric()},
        logdir=args.logdir,
    )

    logs = model.fit(train, dev=dev, epochs=args.epochs)

    # Return all metrics for ReCodEx to validate.
    return {metric: value for metric, value in logs.items()}


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    main(main_args)
