#!/usr/bin/env python3
# Group IDs: 31ff17c9-b0b8-449e-b0ef-8a1aa1e14eb3, 5b78caaa-8040-46f7-bf54-c13e183bbbf8

import argparse
import datetime
import os
import re

import torch
import torch.nn
import torch.nn.functional as F

import npfl138
from npfl138.datasets.morpho_dataset import MorphoDataset

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--batch_size", default=10, type=int, help="Batch size.")
parser.add_argument("--epochs", default=5, type=int, help="Number of epochs.")
parser.add_argument("--max_sentences", default=None, type=int, help="Maximum number of sentences to load.")
parser.add_argument("--recodex", default=False, action="store_true", help="Evaluation in ReCodEx.")
parser.add_argument("--rnn", default="LSTM", choices=["LSTM", "GRU"], help="RNN layer type.")
parser.add_argument("--rnn_dim", default=64, type=int, help="RNN layer dimension.")
parser.add_argument("--seed", default=41, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
parser.add_argument("--we_dim", default=128, type=int, help="Word embedding dimension.")
# If you add more arguments, ReCodEx will keep them with your default values.
parser.add_argument("--logdir", default="logs", type=str, help="Logging directory.")


class Model(npfl138.TrainableModule):
    def __init__(self, args: argparse.Namespace, train: MorphoDataset.Dataset) -> None:
        super().__init__()
        # Create all needed layers.
        # TODO(tagger_we): Create a `torch.nn.Embedding` layer, embedding the word ids
        # from `train.words.string_vocab` to dimensionality `args.we_dim`.
        
        self._pad_id = MorphoDataset.PAD
        self._word_embedding = torch.nn.Embedding(
            num_embeddings=len(train.words.string_vocab),
            embedding_dim=args.we_dim,
            padding_idx=self._pad_id,
        )

        # TODO(tagger_we): Create an RNN layer, either `torch.nn.LSTM` or `torch.nn.GRU` depending
        # on `args.rnn`. The layer should be bidirectional (`bidirectional=True`) with
        # dimensionality `args.rnn_dim`. During the model computation, the layer will
        # process the word embeddings generated by the `self._word_embedding` layer,
        # and we will sum the outputs of forward and backward directions.
        
        rnn_class = torch.nn.LSTM if args.rnn == "LSTM" else torch.nn.GRU
        self._word_rnn = rnn_class(
            args.we_dim,
            args.rnn_dim,
            bidirectional=True,
            batch_first=True,
        )
        # TODO(tagger_we): Create an output linear layer (`torch.nn.Linear`) processing the RNN output,
        # producing logits for tag prediction; `train.tags.string_vocab` is the tag vocabulary.
        
        self._output_layer = torch.nn.Linear(
            args.rnn_dim,
            len(train.tags.string_vocab),
        )

    def forward(self, word_ids: torch.Tensor) -> torch.Tensor:
        # TODO(tagger_we): Start by embedding the `word_ids` using the word embedding layer.
        hidden = self._word_embedding(word_ids)

        # TODO(tagger_we): Process the embedded words through the RNN layer. Because the sentences
        # have different length, you have to use `torch.nn.utils.rnn.pack_padded_sequence`
        # to construct a variable-length `PackedSequence` from the input. You need to compute
        # the length of each sentence in the batch (by counting non-`MorphoDataset.PAD` tokens);
        # note that these lengths must be on CPU, so you might need to use the `.cpu()` method.
        # Finally, also pass `batch_first=True` and `enforce_sorted=False` to the call.
        lengths = (word_ids != self._pad_id).sum(dim=1).cpu()
        packed = torch.nn.utils.rnn.pack_padded_sequence(
            hidden, lengths, batch_first=True, enforce_sorted=False
        )

        # TODO(tagger_we): Pass the `PackedSequence` through the RNN, choosing the appropriate output.
        packed, _ = self._word_rnn(packed)

        # TODO(tagger_we): Unpack the RNN output using the `torch.nn.utils.rnn.pad_packed_sequence` with
        # `batch_first=True` argument. Then sum the outputs of forward and backward directions.
        
        unpacked, _ = torch.nn.utils.rnn.pad_packed_sequence(packed, batch_first=True)
        hidden = unpacked[:, :, :self._word_rnn.hidden_size] + unpacked[:, :, self._word_rnn.hidden_size:]

        # TODO(tagger_we): Pass the RNN output through the output layer. Such an output has a shape
        # `[batch_size, sequence_length, num_tags]`, but the loss and the metric expect
        # the `num_tags` dimension to be in front (`[batch_size, num_tags, sequence_length]`),
        # so you need to reorder the dimensions.
        hidden = self._output_layer(hidden)

        return hidden.permute(0, 2, 1)

    def compute_loss(self, y_pred: torch.Tensor, y_true: torch.Tensor, word_ids: torch.Tensor) -> torch.Tensor:
        # TODO: Compute the loss as the negative log-likelihood of the gold data `y_true`.
        # The computation must process the whole batch at once.
        # - Start by computing the log probabilities of the predictions using the `log_softmax` method.
        # - Compute the alphas according to the CTC algorithm.
        # - Then, you need to take the appropriate alpha for every batch example (using the corresponding
        #   lengths of `y_pred` and also `y_true`) and compute the loss from it.
        # - The losses of the individual examples should be divided by the length of the
        #   target sequence (excluding padding; use 1 if the target sequence is empty).
        #   - This is because we want to compute averages per logit; the `torch.nn.CTCLoss` does
        #     exactly the same when `reduction="mean"` (the default) is used.
        # - Finally, return the mean of the resulting losses.
        #
        # Several comments:
        # - You can add numbers represented in log space using `torch.logsumexp`/`torch.logaddexp`.
        # - With a slight abuse of notation, use `MorphoDataset.PAD` as the blank label in the CTC algorithm
        #   because `MorphoDataset.PAD` is never a valid output tag.
        # - During the computation, I use `-1e9` as the representation of negative infinity; using
        #   `-torch.inf` did not work for me because some operations were not well defined.
        # - During the loss computation, in some cases the target sequence cannot be produced at all.
        #   In that case return 0 as the loss (the same behaviour as passing `zero_infinity=True`
        #   to `torch.nn.CTCLoss`).
        # - During development, you can compare your outputs to the outputs of `torch.nn.CTCLoss`
        #   (with `reduction="none"` you can compare individual batch examples; in that case,
        #   the normalization by the target sequence lengths is not performed).  However, in ReCodEx,
        #   `torch.nn.CTCLoss` is not available.
        log_probs = y_pred.log_softmax(1)

        input_lengths = (word_ids != MorphoDataset.PAD).sum(dim=1).cpu()
        target_lengths = (y_true != MorphoDataset.PAD).sum(dim=1).cpu()

        batch_size, _, max_len_pred = log_probs.shape
        max_len_true = y_true.shape[1]
        
        neg_inf = -1e9
        alpha_blank = torch.full((batch_size, max_len_true + 1, max_len_pred), neg_inf, device=y_pred.device)
        alpha_non_blank = torch.full((batch_size, max_len_true + 1, max_len_pred), neg_inf, device=y_pred.device)
        
        # Initialization according to the algorithm
        alpha_blank[:, 0, 0] = log_probs[:, 0, MorphoDataset.PAD]
        alpha_non_blank[:, 1, 0] = log_probs[:, :, 0].gather(1, y_true[:, 0].reshape((-1, 1))).squeeze()

        target_validation = (~y_true.diff(prepend=-torch.ones((batch_size, 1), device=y_true.device)).to(torch.bool)).to(torch.int)

        for t in range(1, max_len_pred):
            alpha_non_blank[:, 1:, t] = log_probs[:, :, t].gather(1, y_true)
            alpha_non_blank[:, 1:, t] += torch.logsumexp(torch.stack([
                alpha_non_blank[:, 1:, t - 1],
                alpha_blank[:, :-1, t - 1],
                alpha_non_blank[:, :-1, t - 1] + neg_inf * target_validation], dim=1), dim=1)
            
            alpha_blank[:, :, t] = torch.logsumexp(torch.stack([
                alpha_non_blank[:, :, t - 1], 
                alpha_blank[:, :, t - 1]],dim=1),dim=1) + log_probs[:, MorphoDataset.PAD, t].reshape(-1, 1)

        device = y_pred.device
        input_lengths = input_lengths.to(device)
        target_lengths = target_lengths.to(device)

        last_time_steps = torch.clamp(input_lengths - 1, max=max_len_pred - 1)  
        last_target_positions = target_lengths 

        batch_idx = torch.arange(batch_size, device=device)

        # Gather final alpha values using advanced indexing.
        non_blank_vals = alpha_non_blank[batch_idx, last_target_positions, last_time_steps]
        blank_vals = alpha_blank[batch_idx, last_target_positions, last_time_steps]
        final_alphas = torch.stack([non_blank_vals, blank_vals], dim=1)

        # Compute the per-example loss as the negative log-sum-exp over the two values.
        losses = torch.logsumexp(final_alphas, dim=1)

        # Compute final loss per example, normalized by the target length (clamping to at least 1)
        losses = -losses / target_lengths.float().clamp(min=1)
        losses = torch.where(losses > 1e7, torch.zeros_like(losses), losses)
        
        return losses.mean()

    def ctc_decoding(self, logits: torch.Tensor, word_ids: torch.Tensor) -> list[torch.Tensor]:
        # TODO: Implement greedy CTC decoding algorithm. The result should be a list of
        # decoded tag sequences, each sequence (batch example) with appropriate length
        # (i.e., at this point we do not pad the predictions in the batch to the same length).
        #
        # The greedy algorithm should, for every batch example:
        # - consider only the predictions corresponding to valid words (i.e., not the padding ones);
        # - compute the most probable extended label for every one of them;
        # - remove repeated labels;
        # - finally remove the blank labels (which are `MorphoDataset.PAD` in our case).
        
        input_lengths = (word_ids != self._pad_id).sum(dim=1)
        batch_size = logits.size(0)
        predictions = []
        for i in range(batch_size):
            T_i = input_lengths[i].item()
            if T_i == 0:
                predictions.append(torch.tensor([], device=logits.device, dtype=torch.long))
                continue
            logits_i = logits[i, :, :T_i].argmax(dim=0)
            collapsed = []
            prev = None
            for tag in logits_i:
                if tag != prev:
                    if tag != self._pad_id:
                        collapsed.append(tag)
                    prev = tag
            predictions.append(torch.tensor(collapsed, device=logits.device, dtype=torch.long))
        return predictions

    def compute_metrics(self, y_pred: torch.Tensor, y_true: torch.Tensor, word_ids: torch.Tensor) -> dict[str, torch.Tensor]:
        # TODO: Compute predictions using the `ctc_decoding`.
        predictions = self.ctc_decoding(y_pred, word_ids)
        self.metrics["edit_distance"].update(predictions, y_true)
        return {name: metric.compute() for name, metric in self.metrics.items()}

    def predict_step(self, xs, as_numpy=True):
        with torch.no_grad():
            batch = self.ctc_decoding(self.forward(*xs), *xs)
            if as_numpy:
                batch = [example.numpy(force=True) for example in batch]
            return batch


class TrainableDataset(npfl138.TransformedDataset):
    def transform(self, example):
        # TODO(tagger_we): Construct a single example, each consisting of the following pair:
        # - a PyTorch tensor of integer ids of input words as input,
        # - a PyTorch tensor of integer tag ids as targets.
        # To create the ids, use `string_vocab` of `self.dataset.words` and `self.dataset.tags`.
        #
        # TODO: However, compared to `tagger_we`, keep in the target sequence only the tags
        # starting with "B-" (before remapping them to ids).
        word_ids = torch.tensor(
            [self.dataset.words.string_vocab.index(word) for word in example["words"]],
            dtype=torch.long,
        )
        tag_ids = torch.tensor([
            self.dataset.tags.string_vocab.index(tag)
            for tag in example["tags"] if tag.startswith("B-")
        ], dtype=torch.long)
        return word_ids, tag_ids

    def collate(self, batch):
        word_ids, tag_ids = zip(*batch)
        # TODO(tagger_we): Combine `word_ids` into a single tensor, padding shorter
        # sequences to length of the longest sequence in the batch with zeros
        # using `torch.nn.utils.rnn.pad_sequence` with `batch_first=True` argument.
        word_ids = torch.nn.utils.rnn.pad_sequence(word_ids, batch_first=True, padding_value=MorphoDataset.PAD)
        # TODO(tagger_we): Process `tag_ids` analogously to `word_ids`.
        tag_ids = torch.nn.utils.rnn.pad_sequence(tag_ids, batch_first=True, padding_value=MorphoDataset.PAD)
        return word_ids, tag_ids


def main(args: argparse.Namespace) -> dict[str, float]:
    # Set the random seed and the number of threads.
    npfl138.startup(args.seed, args.threads)
    npfl138.global_keras_initializers()

    # Create logdir name.
    args.logdir = os.path.join(args.logdir, "{}-{}".format(
        os.path.basename(globals().get("__file__", "notebook")),
        datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
    ))

    # Load the data.
    morpho = MorphoDataset("czech_cnec", max_sentences=args.max_sentences)
    
    # Prepare the data for training.
    train = TrainableDataset(morpho.train).dataloader(batch_size=args.batch_size, shuffle=True)
    dev = TrainableDataset(morpho.dev).dataloader(batch_size=args.batch_size)
    
    # Create the model and train.
    model = Model(args, morpho.train)
    model.configure(
        # TODO(tagger_we): Create the Adam optimizer.
        optimizer=torch.optim.Adam(model.parameters()),
        # We compute the loss using `compute_loss` method, so no `loss` is passed here.
        metrics={
            # TODO: Create `npfl138.metrics.EditDistance` evaluating CTC greedy decoding, passing
            # `ignore_index=morpho.PAD`.
            "edit_distance": npfl138.metrics.EditDistance(ignore_index=MorphoDataset.PAD)
        },
        logdir=args.logdir,
    )
    logs = model.fit(train, dev=dev, epochs=args.epochs)

    # Return development metrics for ReCodEx to validate.
    return {metric: value for metric, value in logs.items()}


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    main(main_args)