#!/usr/bin/env python3
# Group IDs: 31ff17c9-b0b8-449e-b0ef-8a1aa1e14eb3, 5b78caaa-8040-46f7-bf54-c13e183bbbf8

import argparse
import datetime
import os

import torch
import torchmetrics

import npfl138

npfl138.require_version("2425.9")
from npfl138.datasets.morpho_dataset import MorphoDataset
from npfl138.datasets.morpho_analyzer import MorphoAnalyzer

from lemmatizer_attn import Model, TrainableDataset

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=64, type=int, help="Batch size.")
parser.add_argument(
    "--cle_dim", default=256, type=int, help="Character-level embedding dimension."
)
parser.add_argument(
    "--rnn_dim", default=256, type=int, help="RNN hidden state dimension."
)
parser.add_argument(
    "--tie_embeddings",
    default=True,
    action="store_true",
    help="Tie target embeddings to output weights (requires cle_dim == rnn_dim).",
)
parser.add_argument(
    "--show_results_every_batch",
    default=100,
    type=int,
    help="Print intermediate lemmatization examples every N batches (0 to disable).",
)
parser.add_argument("--epochs", default=20, type=int, help="Number of training epochs.")
parser.add_argument(
    "--learning_rate",
    default=5e-4,
    type=float,
    help="Learning rate for Adam optimizer.",
)
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument(
    "--threads", default=1, type=int, help="Maximum number of threads to use."
)
parser.add_argument(
    "--max_sentences",
    default=None,
    type=int,
    help="Maximum number of sentences to load (for quick experiments).",
)


def main(args: argparse.Namespace) -> None:
    device=torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set random seed and number of threads
    npfl138.startup(args.seed, args.threads)
    npfl138.global_keras_initializers()

    # Create unique log directory
    args.logdir = os.path.join(
        "logs",
        "{}-{}-{}".format(
            os.path.basename(globals().get("__file__", "notebook")),
            datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
            ",".join(f"{k}={v}" for k, v in sorted(vars(args).items())),
        ),
    )

    # Load the Czech PDT data and analyses
    morpho = MorphoDataset("czech_pdt", max_sentences=args.max_sentences)
    analyses = MorphoAnalyzer("czech_pdt_analyses") 

    # Prepare data loaders
    train_loader = TrainableDataset(morpho.train, training=True).dataloader(
        batch_size=args.batch_size, shuffle=True
    )
    dev_loader = TrainableDataset(morpho.dev, training=False).dataloader(
        batch_size=args.batch_size
    )
    test_loader = TrainableDataset(morpho.test, training=False).dataloader(
        batch_size=args.batch_size
    )

    # Initialize and configure the model
    model = Model(args, morpho.train).to(device)
    model.configure(
        optimizer=torch.optim.Adam(model.parameters(), lr=args.learning_rate),
        loss=torch.nn.CrossEntropyLoss(ignore_index=MorphoDataset.PAD),
        metrics={"accuracy": torchmetrics.MeanMetric()},
        logdir=args.logdir,
    )

    # Train the model
    model.fit(train_loader, dev=dev_loader, epochs=args.epochs)

    # Generate test set predictions
    os.makedirs(args.logdir, exist_ok=True)
    output_path = os.path.join(args.logdir, "lemmatizer_competition.txt")
    with open(output_path, "w", encoding="utf-8") as predictions_file:
        preds_iter = iter(model.predict(test_loader, data_with_labels=True))
        for sentence in morpho.test.words.strings:
            for word in sentence:
                # Neural prediction
                neural_idx = next(preds_iter)
                neural_str = "".join(morpho.test.lemmas.char_vocab.strings(neural_idx))
                # Analyzer fallback
                lemma_str = neural_str
                try:
                    analyses_list = analyses.analyze(word)
                    if analyses_list:
                        first = analyses_list[0]
                        lemma_str = (
                            first.lemma if hasattr(first, "lemma") else str(first)
                        )
                except Exception:
                    pass
                print(lemma_str, file=predictions_file)
            print(file=predictions_file)


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    main(main_args)
