#!/usr/bin/env python3
import argparse
import datetime
import os
import re

import numpy as np
import torch
import torchmetrics
import transformers

import npfl138
npfl138.require_version("2425.10")
from npfl138.datasets.text_classification_dataset import TextClassificationDataset

DEVICE = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

# TODO: Define reasonable defaults and optionally more parameters.
# Also, you can set the number of threads to 0 to use all your CPU cores.
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=32, type=int, help="Batch size.")
parser.add_argument("--epochs", default=3, type=int, help="Number of epochs.")
parser.add_argument("--lr", default=5e-5, type=float, help="Learning rate.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")


class Model(npfl138.TrainableModule):
    def __init__(self, args: argparse.Namespace, eleczech: transformers.PreTrainedModel,
                 dataset: TextClassificationDataset.Dataset) -> None:
        super().__init__()

        # TODO: Define the model. Note that
        # - the dimension of the EleCzech output is `eleczech.config.hidden_size`;
        # - the size of the vocabulary of the output labels is `len(dataset.label_vocab)`.
        self.eleczech = eleczech
        hidden_size = eleczech.config.hidden_size
        num_labels = len(dataset.label_vocab)
        self.dropout = torch.nn.Dropout(0.1)
        self.classifier = torch.nn.Linear(hidden_size, num_labels)

    # TODO: Implement the model computation.
    def forward(self, input_ids: torch.Tensor,
                attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.eleczech(input_ids=input_ids,
                                attention_mask=attention_mask)
        cls_rep = outputs.last_hidden_state[:, 0]
        logits = self.classifier(self.dropout(cls_rep))
        return logits


class TrainableDataset(npfl138.TransformedDataset):
    def __init__(self, dataset: TextClassificationDataset.Dataset,
                 tokenizer: transformers.PreTrainedTokenizer) -> None:
        super().__init__(dataset)
        self.tokenizer = tokenizer

    def transform(self, example):
        # TODO: Process single examples containing `example["document"]` and `example["label"]`.
        text = example['document']
        label_id = self.dataset.label_vocab.indices([example['label']])[0]
        return text, label_id

    def collate(self, batch):
        # TODO: Construct a single batch using a list of examples from the `transform` function.
        texts, labels = zip(*batch)
        encoded = self.tokenizer(
            list(texts), padding='longest', truncation=True, return_tensors='pt'
        )
        input_ids = encoded.input_ids.to(DEVICE)
        attention_mask = encoded.attention_mask.to(DEVICE)
        labels = torch.tensor(labels, dtype=torch.long, device=DEVICE)
        return (input_ids, attention_mask), labels


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

    # Load the Electra Czech small lowercased.
    tokenizer = transformers.AutoTokenizer.from_pretrained("ufal/eleczech-lc-small")
    eleczech = transformers.AutoModel.from_pretrained("ufal/eleczech-lc-small").to(DEVICE)

    # Load the data.
    facebook = TextClassificationDataset("czech_facebook")

    # TODO: Prepare the data for training.
    train_loader = TrainableDataset(facebook.train, tokenizer).dataloader(
        batch_size=args.batch_size, shuffle=True
    )
    dev_loader = TrainableDataset(facebook.dev, tokenizer).dataloader(
        batch_size=args.batch_size
    )

    # Create the model.
    model = Model(args, eleczech, facebook.train).to(DEVICE)

    # TODO: Configure and train the model
    model.configure(
        optimizer=torch.optim.AdamW(model.parameters(), lr=args.lr),
        loss=torch.nn.CrossEntropyLoss(),
        metrics={'accuracy': torchmetrics.Accuracy(task='multiclass', num_classes=len(facebook.train.label_vocab)).to(DEVICE)}
    )
    
    model.fit(train_loader, dev=dev_loader, epochs=args.epochs)

    test_docs = facebook.test.data['documents']
    all_preds = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(test_docs), args.batch_size):
            batch_texts = test_docs[i:i+args.batch_size]
            encoded = tokenizer(
                batch_texts, padding='longest', truncation=True, return_tensors='pt'
            )
            input_ids = encoded.input_ids.to(DEVICE)
            attention_mask = encoded.attention_mask.to(DEVICE)
            logits = model(input_ids, attention_mask)
            batch_preds = torch.argmax(logits, dim=1).tolist()
            all_preds.extend(batch_preds)
    
    # Generate test set annotations, but in `args.logdir` to allow parallel execution.
    os.makedirs(args.logdir, exist_ok=True)
    with open(os.path.join(args.logdir, "sentiment_analysis.txt"), "w", encoding="utf-8") as predictions_file:
        for idx in all_preds:
            predictions_file.write(facebook.test.label_vocab.string(idx) + '\n')


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    main(main_args)
