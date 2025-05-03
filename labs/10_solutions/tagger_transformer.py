#!/usr/bin/env python3
import argparse
import datetime
import os
import re
import math

import torch
import torchmetrics

import npfl138

npfl138.require_version("2425.10")
from npfl138.datasets.morpho_dataset import MorphoDataset

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--batch_size", default=10, type=int, help="Batch size.")
parser.add_argument("--epochs", default=5, type=int, help="Number of epochs.")
parser.add_argument(
    "--max_sentences",
    default=None,
    type=int,
    help="Maximum number of sentences to load.",
)
parser.add_argument(
    "--recodex", default=False, action="store_true", help="Evaluation in ReCodEx."
)
parser.add_argument(
    "--transformer_dropout", default=0.0, type=float, help="Transformer dropout."
)
parser.add_argument(
    "--transformer_expansion",
    default=4,
    type=int,
    help="Transformer FFN expansion factor.",
)
parser.add_argument(
    "--transformer_heads", default=4, type=int, help="Transformer heads."
)
parser.add_argument(
    "--transformer_layers", default=2, type=int, help="Transformer layers."
)
parser.add_argument("--seed", default=46, type=int, help="Random seed.")
parser.add_argument(
    "--threads", default=1, type=int, help="Maximum number of threads to use."
)
parser.add_argument("--we_dim", default=64, type=int, help="Word embedding dimension.")
# If you add more arguments, ReCodEx will keep them with your default values.


class Model(npfl138.TrainableModule):
    class FFN(torch.nn.Module):
        def __init__(self, dim: int, expansion: int) -> None:
            super().__init__()
            # TODO: Create the required layers -- first a ReLU-activated dense
            # layer with `dim * expansion` units, followed by a dense layer
            # with `dim` units without an activation.
            # first a ReLU-activated dense layer, then a dense layer back to dim
            self.linear1 = torch.nn.Linear(dim, dim * expansion)
            self.linear2 = torch.nn.Linear(dim * expansion, dim)

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            # TODO: Execute the FFN Transformer layer.
            x = self.linear1(inputs)
            x = torch.relu(x)
            x = self.linear2(x)
            return x

    class SelfAttention(torch.nn.Module):
        def __init__(self, dim: int, heads: int) -> None:
            super().__init__()
            self.dim, self.heads = dim, heads
            # TODO: Create weight matrices W_Q, W_K, W_V, and W_O; each a module parameter
            # `torch.nn.Parameter` of shape `[dim, dim]`. The weights should be initialized using
            # the `torch.nn.init.xavier_uniform_` in the same order the matrices are listed above.

            self.W_Q = torch.nn.Parameter(torch.empty(dim, dim))
            self.W_K = torch.nn.Parameter(torch.empty(dim, dim))
            self.W_V = torch.nn.Parameter(torch.empty(dim, dim))
            self.W_O = torch.nn.Parameter(torch.empty(dim, dim))
            
            torch.nn.init.xavier_uniform_(self.W_Q)
            torch.nn.init.xavier_uniform_(self.W_K)
            torch.nn.init.xavier_uniform_(self.W_V)
            torch.nn.init.xavier_uniform_(self.W_O)

        def forward(self, inputs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
            # TODO: Execute the self-attention layer.
            #
            # Start by computing Q, K and V. In all cases:
            # - first multiply `inputs` by the corresponding weight matrix W_Q/W_K/W_V,
            # - reshape via `torch.reshape` to `[batch_size, max_sentence_len, heads, dim // heads]`,
            # - permute dimensions via `torch.permute` to `[batch_size, heads, max_sentence_len, dim // heads]`.

            # TODO: Continue by computing the self-attention weights as Q @ K^T,
            # normalizing by the square root of `dim // heads`.

            # TODO: Apply the softmax, but including a suitable mask ignoring all padding words.
            # The original `mask` is a bool matrix of shape `[batch_size, max_sentence_len]`
            # indicating which words are valid (nonzero value) or padding (zero value).
            # To mask an input to softmax, replace it by -1e9 (theoretically we should use
            # minus infinity, but `torch.exp(-1e9)` is also zero because of limited precision).

            # TODO: Finally,
            # - take a weighted combination of values V according to the computed attention
            #   (using a suitable matrix multiplication),
            # - permute the result to `[batch_size, max_sentence_len, heads, dim // heads]`,
            # - reshape to `[batch_size, max_sentence_len, dim]`,
            # - multiply the result by the W_O matrix.

            batch, seq_len, dim = inputs.size()
            head_dim = dim // self.heads
            # Linear projections
            Q = inputs.matmul(self.W_Q)
            K = inputs.matmul(self.W_K)
            V = inputs.matmul(self.W_V)
            # Reshape and permute to [batch, heads, seq_len, head_dim]
            Q = Q.view(batch, seq_len, self.heads, head_dim).permute(0, 2, 1, 3)
            K = K.view(batch, seq_len, self.heads, head_dim).permute(0, 2, 1, 3)
            V = V.view(batch, seq_len, self.heads, head_dim).permute(0, 2, 1, 3)
            # Scaled dot-product attention
            scores = Q.matmul(K.transpose(-2, -1)) / (head_dim ** 0.5)
            # Mask padding tokens
            # mask: [batch, seq_len] -> [batch, 1, 1, seq_len]
            mask_k = mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(~mask_k, -1e9)
            attn = torch.softmax(scores, dim=-1)
            # Weighted sum of values
            context = attn.matmul(V)  # [batch, heads, seq_len, head_dim]
            # Reassemble
            context = context.permute(0, 2, 1, 3).contiguous().view(batch, seq_len, dim)
            # Final linear
            output = context.matmul(self.W_O)
            return output

    class PositionalEmbedding(torch.nn.Module):
        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            # TODO: Compute the sinusoidal positional embeddings. Assuming the embeddings have
            # a shape `[max_sentence_len, dim]` with `dim` even, and for `0 <= i < dim/2`:
            # - the value on index `[pos, i]` should be
            #     `sin(pos / 10_000 ** (2 * i / dim))`
            # - the value on index `[pos, dim/2 + i]` should be
            #     `cos(pos / 10_000 ** (2 * i / dim))`
            # - the `0 <= pos < max_sentence_len` is the sentence index.
            # This order is the same as in the visualization on the slides, but
            # different from the original paper where `sin` and `cos` interleave.

            # inputs: [batch, seq_len, dim]
            seq_len, dim = inputs.size(1), inputs.size(2)
            device = inputs.device
            # Compute positional encodings
            position = torch.arange(seq_len, dtype=torch.float, device=device).unsqueeze(1)
            div_term = torch.exp(
                torch.arange(0, dim // 2, dtype=torch.float, device=device) *
                (-math.log(10000.0) * 2 / dim)
            )
            pe_sin = torch.sin(position * div_term)
            pe_cos = torch.cos(position * div_term)
            pe = torch.cat([pe_sin, pe_cos], dim=1)
            return pe.unsqueeze(0)

    class Transformer(torch.nn.Module):
        def __init__(
            self, layers: int, dim: int, expansion: int, heads: int, dropout: float
        ) -> None:
            super().__init__()
            # TODO: Create:
            # - the positional embedding layer;
            # - the required number of transformer layers, each consisting of
            #   - a layer normalization and a self-attention layer followed by a dropout layer,
            #   - a layer normalization and a FFN layer followed by a dropout layer.
            # During ReCodEx evaluation, the order of layer creation is not important,
            # but if you want to get the same results as on the course website, create
            # the layers in the order they are called in the `forward` method.

            self.positional = Model.PositionalEmbedding()
            self.norm1_layers = torch.nn.ModuleList()
            self.attn_layers = torch.nn.ModuleList()
            self.drop1_layers = torch.nn.ModuleList()
            self.norm2_layers = torch.nn.ModuleList()
            self.ffn_layers = torch.nn.ModuleList()
            self.drop2_layers = torch.nn.ModuleList()
            for _ in range(layers):
                self.norm1_layers.append(torch.nn.LayerNorm(dim))
                self.attn_layers.append(Model.SelfAttention(dim, heads))
                self.drop1_layers.append(torch.nn.Dropout(dropout))
                self.norm2_layers.append(torch.nn.LayerNorm(dim))
                self.ffn_layers.append(Model.FFN(dim, expansion))
                self.drop2_layers.append(torch.nn.Dropout(dropout))

        def forward(self, inputs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
            # TODO: First compute the positional embeddings.

            # TODO: Add the positional embeddings to the `inputs` and then
            # perform the given number of transformer layers, composed of
            # - a self-attention sub-layer, followed by
            # - a FFN sub-layer.
            # In each sub-layer, pass the input through LayerNorm, then compute
            # the corresponding operation, apply dropout, and finally add this result
            # to the original sub-layer input. Note that the given `mask` should be
            # passed to the self-attention operation to ignore the padding words.

            hidden = inputs + self.positional(inputs)
            for i in range(len(self.attn_layers)):
                # Self-attention sub-layer (Pre-LN)
                h_norm = self.norm1_layers[i](hidden)
                attn_out = self.attn_layers[i](h_norm, mask)
                attn_out = self.drop1_layers[i](attn_out)
                hidden = hidden + attn_out
                # FFN sub-layer (Pre-LN)
                h_norm2 = self.norm2_layers[i](hidden)
                ffn_out = self.ffn_layers[i](h_norm2)
                ffn_out = self.drop2_layers[i](ffn_out)
                hidden = hidden + ffn_out
            return hidden

    def __init__(self, args: argparse.Namespace, train: MorphoDataset.Dataset) -> None:
        super().__init__()

        # Create all needed layers.
        # TODO(tagger_we): Create a `torch.nn.Embedding` layer, embedding the word ids
        # from `train.words.string_vocab` to dimensionality `args.we_dim`.
        self._word_embedding = torch.nn.Embedding(
            num_embeddings=len(train.words.string_vocab),
            embedding_dim=args.we_dim,
            padding_idx=MorphoDataset.PAD
        )

        # TODO: Create a `Model.Transformer` layer, using suitable options from `args`
        #   (using `args.we_dim` for the `dim` argument),
        self._transformer = Model.Transformer(
            layers=args.transformer_layers,
            dim=args.we_dim,
            expansion=args.transformer_expansion,
            heads=args.transformer_heads,
            dropout=args.transformer_dropout
        )

        # TODO(tagger_we): Create an output linear layer (`torch.nn.Linear`) processing the RNN output,
        # producing logits for tag prediction; `train.tags.string_vocab` is the tag vocabulary.
        self._output_layer = torch.nn.Linear(args.we_dim, len(train.tags.string_vocab))


    def forward(self, word_ids: torch.Tensor) -> torch.Tensor:
        # TODO(tagger_we): Start by embedding the `word_ids` using the word embedding layer.

        # TODO: Process the embedded words through the transformer. As the second argument,
        # pass the attention mask `word_ids != MorphoDataset.PAD`.

        # TODO(tagger_we): Pass `hidden` through the output layer. Such an output has a shape
        # `[batch_size, sequence_length, num_tags]`, but the loss and the metric expect
        # the `num_tags` dimension to be in front (`[batch_size, num_tags, sequence_length]`),
        # so you need to reorder the dimensions.

        # word_ids: [batch, seq_len]
        mask = word_ids != MorphoDataset.PAD
        hidden = self._word_embedding(word_ids)
        hidden = self._transformer(hidden, mask)
        logits = self._output_layer(hidden)
        # [batch, seq_len, num_tags] -> [batch, num_tags, seq_len]
        return logits.permute(0, 2, 1)


class TrainableDataset(npfl138.TransformedDataset):
    def transform(self, example):
        # TODO(tagger_we): Construct a single example, each consisting of the following pair:
        # - a PyTorch tensor of integer ids of input words as input,
        # - a PyTorch tensor of integer tag ids as targets.
        # To create the ids, use `string_vocab` of `self.dataset.words` and `self.dataset.tags`.
        word_ids = torch.tensor(
            self.dataset.words.string_vocab.indices(example["words"]), dtype=torch.long
        )
        tag_ids = torch.tensor(
            self.dataset.tags.string_vocab.indices(example["tags"]), dtype=torch.long
        )
        return word_ids, tag_ids

    def collate(self, batch):
        # Construct a single batch, where `batch` is a list of examples
        # generated by `transform`.
        word_ids, tag_ids = zip(*batch)
        # TODO(tagger_we): Combine `word_ids` into a single tensor, padding shorter
        # sequences to length of the longest sequence in the batch with zeros
        # using `torch.nn.utils.rnn.pad_sequence` with `batch_first=True` argument.
        word_ids = torch.nn.utils.rnn.pad_sequence(
            word_ids, batch_first=True, padding_value=MorphoDataset.PAD
        )
        # TODO(tagger_we): Process `tag_ids` analogously to `word_ids`.
        tag_ids = torch.nn.utils.rnn.pad_sequence(
            tag_ids, batch_first=True, padding_value=MorphoDataset.PAD
        )
        return word_ids, tag_ids


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
    train = TrainableDataset(morpho.train).dataloader(
        batch_size=args.batch_size, shuffle=True
    )
    dev = TrainableDataset(morpho.dev).dataloader(batch_size=args.batch_size)

    # Create the model and train.
    model = Model(args, morpho.train)

    model.configure(
        # TODO(tagger_we): Create the Adam optimizer.
        optimizer=torch.optim.Adam(model.parameters()),
        # TODO(tagger_we): Use the usual `torch.nn.CrossEntropyLoss` loss function. Additionally,
        # pass `ignore_index=morpho.PAD` to the constructor so that the padded
        # tags are ignored during the loss computation. Note that the loss
        # expects the input to be of shape `[batch_size, num_tags, sequence_length]`.
        loss=torch.nn.CrossEntropyLoss(ignore_index=morpho.PAD),
        # TODO(tagger_we): Create a `torchmetrics.Accuracy` metric, passing "multiclass" as
        # the first argument, `num_classes` set to the number of unique tags, and
        # again `ignore_index=morpho.PAD` to ignore the padded tags.
        metrics={
            "accuracy": torchmetrics.Accuracy(
                task="multiclass",
                num_classes=len(morpho.train.tags.string_vocab),
                ignore_index=morpho.PAD,
            )
        },
        # logdir=args.logdir,
    )

    logs = model.fit(train, dev=dev, epochs=args.epochs)

    # Return development and training losses for ReCodEx to validate.
    return {metric: value for metric, value in logs.items() if "loss" in metric}


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    main(main_args)
