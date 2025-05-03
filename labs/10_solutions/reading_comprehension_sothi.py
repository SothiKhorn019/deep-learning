#!/usr/bin/env python3
# Group IDs: 31ff17c9-b0b8-449e-b0ef-8a1aa1e14eb3, 5b78caaa-8040-46f7-bf54-c13e183bbbf8

import argparse
import datetime
import os
import re

import torch
import torchmetrics
import transformers
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    RobertaForQuestionAnswering,
    get_linear_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
)

import npfl138

npfl138.require_version("2425.10")
from npfl138.datasets.reading_comprehension_dataset import ReadingComprehensionDataset

# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# TODO: Define reasonable defaults and optionally more parameters.
# Also, you can set the number of threads to 0 to use all your CPU cores.
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=8, type=int, help="Batch size.")
parser.add_argument("--learning_rate", default=1e-5, type=float)
parser.add_argument("--max_length", default=384, type=int)
# parser.add_argument("--max_length", default=512, type=int)
parser.add_argument("--epochs", default=8, type=int, help="Number of epochs.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument(
    "--threads", default=1, type=int, help="Maximum number of threads to use."
)


class QAEMetric(torchmetrics.Metric):
    """
    Exact‐match metric for QA: 1.0 if both start and end are exactly correct, else 0.0.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, y_pred, y_true):
        # y_pred: (start_logits, end_logits), y_true: (start_pos, end_pos)
        start_logits, end_logits = y_pred
        start_pos, end_pos = y_true
        pred_start = torch.argmax(start_logits, dim=-1)
        pred_end = torch.argmax(end_logits, dim=-1)
        em = ((pred_start == start_pos) & (pred_end == end_pos)).long()
        self.correct += em.sum()
        self.total += em.numel()

    def compute(self):
        return self.correct.float() / self.total


class Model(npfl138.TrainableModule):
    """
    Wraps RobertaForQuestionAnswering into TrainableModule with explicit loss method.
    """

    def __init__(
        self, args: argparse.Namespace, base_model: RobertaForQuestionAnswering
    ) -> None:
        super().__init__()
        self.qa_model = base_model
        # separate CE loss for start and end
        self.start_loss_fn = torch.nn.CrossEntropyLoss()
        self.end_loss_fn = torch.nn.CrossEntropyLoss()

    def forward(self, input_ids, attention_mask):
        outputs = self.qa_model(
            input_ids=input_ids, attention_mask=attention_mask, return_dict=True
        )
        # return tuple of logits
        return outputs.start_logits, outputs.end_logits

    def compute_qaloss(self, y_pred, y_true):
        # y_pred: (start_logits, end_logits)
        # y_true: (start_positions, end_positions)
        start_logits, end_logits = y_pred
        start_pos, end_pos = y_true
        loss_s = self.start_loss_fn(start_logits, start_pos)
        loss_e = self.end_loss_fn(end_logits, end_pos)
        return loss_s + loss_e


class TrainableDataset(npfl138.TransformedDataset):
    """
    Transforms ReadingComprehensionDataset into (input, label) pairs.
    """

    def __init__(
        self,
        dataset: ReadingComprehensionDataset.Dataset,
        tokenizer: AutoTokenizer,
        max_length: int,
    ):
        super().__init__(dataset)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def transform(self, example):
        context = example["context"]
        qa = example["qas"][0]
        question = qa["question"]
        ans = qa["answers"][0]
        s_char = ans["start"]
        e_char = s_char + len(ans["text"])
        enc = self.tokenizer(
            question,
            context,
            truncation="only_second",
            padding="max_length",
            max_length=self.max_length,
            return_offsets_mapping=True,
        )
        offsets = enc.pop("offset_mapping")
        seq_ids = enc.sequence_ids()
        # default to CLS
        token_start = token_end = enc["input_ids"].index(self.tokenizer.cls_token_id)
        for i, sid in enumerate(seq_ids):
            if sid != 1:
                continue
            cs, ce = offsets[i]
            if cs <= s_char < ce:
                token_start = i
            if cs < e_char <= ce:
                token_end = i
        enc["start_positions"] = token_start
        enc["end_positions"] = token_end
        return enc

    def collate(self, batch):
        input_ids = torch.tensor([b["input_ids"] for b in batch], device=DEVICE)
        attention_mask = torch.tensor(
            [b["attention_mask"] for b in batch], device=DEVICE
        )
        start_pos = torch.tensor([b["start_positions"] for b in batch], device=DEVICE)
        end_pos = torch.tensor([b["end_positions"] for b in batch], device=DEVICE)
        inputs = (input_ids, attention_mask)
        labels = (start_pos, end_pos)
        return inputs, labels


def main(args: argparse.Namespace) -> None:
    npfl138.startup(args.seed, args.threads)
    npfl138.global_keras_initializers()

    # create logdir
    args.logdir = os.path.join(
        "logs",
        f"{os.path.basename(__file__)}-"
        f"{datetime.datetime.now():%Y-%m-%d_%H%M%S}-"
        + ",".join(f"{k}={v}" for k, v in sorted(vars(args).items())),
    )
    os.makedirs(args.logdir, exist_ok=True)

    # load tokenizer & model
    tokenizer = AutoTokenizer.from_pretrained("ufal/robeczech-base")
    base_model = RobertaForQuestionAnswering.from_pretrained(
        "ufal/robeczech-base", 
        ignore_mismatched_sizes=True,
        hidden_dropout_prob=0.2,  # Increase dropout
        attention_probs_dropout_prob=0.2,
    ).to(DEVICE)

    # load dataset
    ds = ReadingComprehensionDataset()

    # train_ds = TrainableDataset(ds.train.paragraphs[:500], tokenizer, args.max_length)
    train_ds = TrainableDataset(ds.train.paragraphs, tokenizer, args.max_length)
    dev_ds = TrainableDataset(ds.dev.paragraphs, tokenizer, args.max_length)

    train_loader = train_ds.dataloader(batch_size=args.batch_size, shuffle=True)
    dev_loader = dev_ds.dataloader(batch_size=args.batch_size)

    # wrap in TrainableModule
    model = Model(args, base_model).to(DEVICE)
    qa_em = QAEMetric().to(DEVICE)

    total_steps = len(train_loader) * args.epochs
    # warmup_steps = int(0.1 * total_steps)
    warmup_steps = int(0.2 * total_steps)
    

    # configure training
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=0.01
    )

    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    # scheduler = get_cosine_schedule_with_warmup(
    #     optimizer,
    #     num_warmup_steps=warmup_steps,
    #     num_training_steps=total_steps,
    #     # num_cycles=0.5,  # half‐cycle cosine
    # )
    model.configure(
        optimizer=optimizer,
        loss=model.compute_qaloss,
        scheduler=scheduler,
        metrics={"accuracy": qa_em},
        logdir=args.logdir,
    ).to(DEVICE),

    model.fit(train_loader, dev=dev_loader, epochs=args.epochs)

    test_preds = []
    model.eval()
    with torch.no_grad():
        for para in ds.test.paragraphs:
            context = para["context"]
            for qa in para["qas"]:
                enc = tokenizer(
                    qa["question"],
                    context,
                    truncation="only_second",
                    padding=False,
                    max_length=args.max_length,
                    return_offsets_mapping=True,
                )
                input_ids = torch.tensor(enc["input_ids"]).unsqueeze(0).to(DEVICE)
                attention_mask = (
                    torch.tensor(enc["attention_mask"]).unsqueeze(0).to(DEVICE)
                )
                start_logits, end_logits = model(input_ids, attention_mask)
                s_idx = int(torch.argmax(start_logits, dim=-1))
                e_idx = int(torch.argmax(end_logits, dim=-1))
                if e_idx < s_idx:
                    e_idx = s_idx
                offsets = enc["offset_mapping"]
                s_char = offsets[s_idx][0]
                e_char = offsets[e_idx][1]
                test_preds.append(context[s_char:e_char])

    # write output
    out_file = os.path.join(args.logdir, "reading_comprehension.txt")
    with open(out_file, "w", encoding="utf-8") as f:
        for ans in test_preds:
            f.write(ans + "\n")
    print(f"Test predictions saved to {args.logdir}")


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    main(main_args) 
