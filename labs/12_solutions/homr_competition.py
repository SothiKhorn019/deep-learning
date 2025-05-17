#!/usr/bin/env python3
import argparse
import datetime
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils

import npfl138
npfl138.require_version("2425.12")
from npfl138.datasets.homr_dataset import HOMRDataset

# Import TrainableModule and TransformedDataset
from npfl138.trainable_module import TrainableModule
from npfl138.transformed_dataset import TransformedDataset

def parse_args():
    parser = argparse.ArgumentParser(description="HOMR competition: handwritten optical music recognition")
    parser.add_argument("--max_samples", default=None, type=int,
                        help="Limit training samples for quick testing.")
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--epochs", default=10, type=int)
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--threads", default=4, type=int)
    parser.add_argument("--logdir", default=None, type=str)
    parser.add_argument("--decode_on_demand", action="store_true")
    parser.add_argument("--device", default="mps", choices=["cpu","mps","cuda"], help="Device to use (mps if available, else cuda).")
    args = parser.parse_args()
    if args.logdir is None:
        args.logdir = os.path.join("logs", f"homr-{datetime.datetime.now():%Y%m%d_%H%M%S}")
    return args


class HOMRTrainableDataset(TransformedDataset):
    """Wrap HOMRDataset examples into (image, marks) pairs and collate for CTC."""
    def transform(self, example):
        img = example["image"]                # [C,H,W] float32 tensor
        marks = torch.tensor(example["marks"], dtype=torch.long)
        return img, marks

    def collate(self, batch):
        xs, ys = zip(*batch)
        # pad images to same H,W
        C = xs[0].shape[0]
        heights = [x.shape[1] for x in xs]
        widths  = [x.shape[2] for x in xs]
        max_h, max_w = max(heights), max(widths)
        xs_padded = torch.zeros(len(xs), C, max_h, max_w)
        for i, x in enumerate(xs):
            c, h, w = x.shape
            xs_padded[i, :, :h, :w] = x
        # input lengths for CTC
        x_lengths = torch.tensor([w // 4 for w in widths], dtype=torch.long)
        # pad targets
        y_lengths = torch.tensor([y.size(0) for y in ys], dtype=torch.long)
        ys_padded = rnn_utils.pad_sequence(ys, batch_first=True, padding_value=HOMRDataset.MARKS)
        return (xs_padded, x_lengths), (ys_padded, y_lengths)


class Model(TrainableModule):
    def __init__(self, args):
        super().__init__()
        self.blank_idx = HOMRDataset.MARKS
        # CNN+BiLSTM
        self.conv = nn.Sequential(
            nn.Conv2d(HOMRDataset.C, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.lstm = nn.LSTM(input_size=64, hidden_size=256, num_layers=2,
                            bidirectional=True)
        self.fc = nn.Linear(256 * 2, HOMRDataset.MARKS + 1)
        self.metrics = {"edit_distance": HOMRDataset.EditDistanceMetric()}

    def forward(self, xs_padded, x_lengths):
        # xs_padded: [B, C, H, W], x_lengths: [B]
        f = self.conv(xs_padded)
        f = f.mean(dim=2)           # [B, 64, W2]
        f = f.permute(2, 0, 1)       # [W2, B, 64]
        out, _ = self.lstm(f)        # [W2, B, 512]
        logits = self.fc(out)        # [W2, B, classes]
        logp = F.log_softmax(logits, dim=2)
        return logp

    def compute_loss(self, y_pred, y_batch, xs_padded, x_lengths):
        ys_padded, y_lengths = y_batch
        # flatten targets
        targets = torch.cat([ys_padded[i, :y_lengths[i]] for i in range(y_lengths.size(0))])
        loss_fn = nn.CTCLoss(blank=self.blank_idx, zero_infinity=True)
        # MPS does not support CTC loss: fallback to CPU
        if y_pred.device.type == 'mps':
            loss = loss_fn(y_pred.cpu(), targets.cpu(), x_lengths.cpu(), y_lengths.cpu())
            return loss.to(y_pred.device)
        return loss_fn(y_pred, targets, x_lengths, y_lengths)

    def compute_loss(self, y_pred, y_batch, xs_padded, x_lengths):
        ys_padded, y_lengths = y_batch
        # flatten targets
        targets = torch.cat([ys_padded[i, :y_lengths[i]] for i in range(y_lengths.size(0))])
        loss_fn = nn.CTCLoss(blank=self.blank_idx, zero_infinity=True)
        # MPS does not support CTC loss: fallback to CPU
        if y_pred.device.type == 'mps':
            loss = loss_fn(y_pred.cpu(), targets.cpu(), x_lengths.cpu(), y_lengths.cpu())
            return loss.to(y_pred.device)
        return loss_fn(y_pred, targets, x_lengths, y_lengths)

    def compute_metrics(self, y_pred, y_batch, xs_padded, x_lengths):
        if not self.training:
            preds = self.ctc_decoding(y_pred, x_lengths)
            pred_str = [" ".join(HOMRDataset.MARK_NAMES[i] for i in seq.tolist()) for seq in preds]
            ys_padded, y_lengths = y_batch
            gold_str = [" ".join(HOMRDataset.MARK_NAMES[int(i)]
                        for i in ys_padded[j, :y_lengths[j]].tolist())
                        for j in range(len(pred_str))]
            self.metrics["edit_distance"].update(pred_str, gold_str)
        return {k: m.compute() for k, m in self.metrics.items()}

    def ctc_decoding(self, y_pred, x_lengths):
        argmax = y_pred.argmax(dim=2)  # [T, B]
        T, B = argmax.shape
        results = []
        for b in range(B):
            seq, prev = [], self.blank_idx
            for t in range(x_lengths[b]):
                idx = int(argmax[t, b])
                if idx != self.blank_idx and idx != prev:
                    seq.append(idx)
                prev = idx
            results.append(torch.tensor(seq, device=y_pred.device))
        return results

    def predict_step(self, batch, as_numpy=True):
        (xs_padded, x_lengths), _ = batch
        xs = xs_padded.to(self.device)
        lengths = x_lengths.to(self.device)
        with torch.no_grad():
            y_pred = self.forward(xs, lengths)
            preds = self.ctc_decoding(y_pred, lengths)
        if as_numpy:
            return [p.cpu().numpy() for p in preds]
        return preds


def main(args):
    # Device selection: prefer MPS, then CUDA, else CPU
    if args.device == 'mps' and torch.backends.mps.is_available():
        device = torch.device('mps')
    elif args.device == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
    elif args.device == 'cpu':
        device = torch.device('cpu')
    else:
        # fallback: use CUDA if available, otherwise CPU
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize random seeds and threading
    npfl138.startup(args.seed, args.threads)

    # Load raw datasets
    raw = HOMRDataset(decode_on_demand=args.decode_on_demand)
    train_ds, dev_ds, test_ds = raw.train, raw.dev, raw.test
    # optional subset
    if args.max_samples is not None:
        train_ds = torch.utils.data.Subset(train_ds, range(min(args.max_samples, len(train_ds))))
    # wrap with transforms and collation
    train_ds = HOMRTrainableDataset(train_ds)
    dev_ds = HOMRTrainableDataset(dev_ds)
    test_ds = HOMRTrainableDataset(test_ds)
    # dataloaders
    train_loader = train_ds.dataloader(batch_size=args.batch_size, shuffle=True)
    dev_loader = dev_ds.dataloader(batch_size=args.batch_size)
    test_loader = test_ds.dataloader(batch_size=args.batch_size)

    # model initialization
    model = Model(args).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    model.configure(optimizer=optimizer, loss=None, metrics=model.metrics, logdir=args.logdir)

    # training
    model.fit(train_loader, dev=dev_loader, epochs=args.epochs)

    # prediction
    preds = []
    for batch in test_loader:
        ((xs_padded, x_lengths), _) = batch
        batch_device = ((xs_padded.to(device), x_lengths.to(device)), batch[1])
        preds.extend(model.predict_step(batch_device, as_numpy=True))

    # write outputs
    os.makedirs(args.logdir, exist_ok=True)
    out_path = os.path.join(args.logdir, "homr_competition.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        for seq in preds:
            f.write(" ".join(HOMRDataset.MARK_NAMES[idx] for idx in seq) + "\n")

if __name__ == "__main__":
    args = parse_args()
    main(args)
