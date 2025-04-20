#!/usr/bin/env python3
# Group IDs: 31ff17c9-b0b8-449e-b0ef-8a1aa1e14eb3, 5b78caaa-8040-46f7-bf54-c13e183bbbf8
import argparse
import datetime
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils

import npfl138
npfl138.require_version("2425.8")
from npfl138.datasets.common_voice_cs import CommonVoiceCs

# Parser configuration
parser = argparse.ArgumentParser()
parser.add_argument("--max_samples", default=None, type=int, help="Limit training samples for quick testing.")
parser.add_argument("--batch_size", default=16, type=int)
parser.add_argument("--epochs", default=10, type=int)
parser.add_argument("--lr", default=1e-3, type=float)
parser.add_argument("--hidden_dim", default=128, type=int)
parser.add_argument("--num_layers", default=2, type=int)
parser.add_argument("--seed", default=42, type=int)
parser.add_argument("--threads", default=12, type=int)


class Model(npfl138.TrainableModule):
    def __init__(self, args, dataset):
        super().__init__()
        self.blank_idx = CommonVoiceCs.PAD
        # acoustic encoder
        self.rnn = nn.GRU(
            input_size=CommonVoiceCs.MFCC_DIM,
            hidden_size=args.hidden_dim,
            num_layers=args.num_layers,
            bidirectional=True,
            batch_first=False
        )
        self.fc = nn.Linear(args.hidden_dim*2, len(CommonVoiceCs.LETTER_NAMES))
        self.metrics = {"edit_distance": CommonVoiceCs.EditDistanceMetric()}

    def forward(self, x, x_lengths):
        # x: [batch, dim, time] -> [time, batch, dim]
        x = x.permute(2,0,1)
        packed = rnn_utils.pack_padded_sequence(x, x_lengths.cpu(), enforce_sorted=False)
        outputs, _ = self.rnn(packed)
        outputs, _ = rnn_utils.pad_packed_sequence(outputs)
        logits = self.fc(outputs)
        return F.log_softmax(logits, dim=2)

    def compute_loss(self, y_pred, y_batch, xs_padded, x_lengths):
        ys_padded, y_lengths = y_batch
        targets = torch.cat([ys_padded[i,:y_lengths[i]] for i in range(y_lengths.size(0))])
        loss_fn = nn.CTCLoss(blank=self.blank_idx, zero_infinity=True)
        if y_pred.device.type=='mps':
            loss = loss_fn(y_pred.cpu(), targets, x_lengths.cpu(), y_lengths.cpu())
            return loss.to(y_pred.device)
        return loss_fn(y_pred, targets, x_lengths, y_lengths)

    def ctc_decoding(self, y_pred, x_lengths):
        # Greedy collapse repeats remove blanks
        # y_pred: [time, batch, classes]
        preds=[]
        argmax = y_pred.argmax(dim=2)  # [time, batch]
        T,B = argmax.shape
        for b in range(B):
            seq = argmax[:,b].tolist()
            collapsed=[]
            prev=None
            for idx in seq:
                if idx!=self.blank_idx and idx!=prev:
                    collapsed.append(idx)
                prev=idx
            preds.append(torch.tensor(collapsed, device=y_pred.device))
        return preds

    def compute_metrics(self, y_pred, y_batch, xs_padded, x_lengths):
        ys_padded, y_lengths = y_batch
        if not self.training:
            preds=self.ctc_decoding(y_pred, x_lengths)
            pred_str=["".join(CommonVoiceCs.LETTER_NAMES[i] for i in seq.tolist()) for seq in preds]
            gold_str=["".join(CommonVoiceCs.LETTER_NAMES[i] for i in ys_padded[j,:y_lengths[j]].tolist()) for j in range(len(preds))]
            self.metrics["edit_distance"].update(pred_str, gold_str)
        return {k:m.compute() for k,m in self.metrics.items()}

    def predict_step(self, batch, as_numpy=True):
        (xs_padded,x_lengths),_ = batch
        with torch.no_grad():
            y_pred=self.forward(xs_padded,x_lengths)
            preds=self.ctc_decoding(y_pred,x_lengths)
        if as_numpy:
            return [p.cpu().numpy() for p in preds]
        return preds

class TrainableDataset(npfl138.TransformedDataset):
    def transform(self, example):
        mfcc=example["mfccs"]
        letters=[CommonVoiceCs.LETTER_NAMES.index(ch) for ch in example["sentence"]]
        return mfcc, torch.tensor(letters,dtype=torch.long)

    def collate(self, batch):
        xs,ys=zip(*batch)
        x_lengths=torch.tensor([x.shape[0] for x in xs],dtype=torch.long)
        xs_padded=rnn_utils.pad_sequence(xs,batch_first=True).permute(0,2,1)
        ys_padded=rnn_utils.pad_sequence(ys,batch_first=True,padding_value=CommonVoiceCs.PAD)
        y_lengths=torch.tensor([y.size(0) for y in ys],dtype=torch.long)
        return (xs_padded,x_lengths),(ys_padded,y_lengths)


def main(args):
    device=torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}",file=sys.stderr)

    npfl138.startup(args.seed,args.threads)
    npfl138.global_keras_initializers()

    args.logdir=os.path.join("logs",f"{os.path.basename(__file__)}-{datetime.datetime.now():%Y-%m-%d_%H%M%S}-"+",".join(f"{k[0]}={v}" for k,v in sorted(vars(args).items())))

    common_voice=CommonVoiceCs()
    
    train_dataset = common_voice.train
    if args.max_samples is not None:
        import torch.utils.data as data_utils
        train_dataset = data_utils.Subset(train_dataset, list(range(min(args.max_samples, len(train_dataset)))))
    
    train_loader=TrainableDataset(train_dataset).dataloader(batch_size=args.batch_size,shuffle=True)
    dev_loader=TrainableDataset(common_voice.dev).dataloader(batch_size=args.batch_size)
    test_loader=TrainableDataset(common_voice.test).dataloader(batch_size=args.batch_size)

    model=Model(args,common_voice.train).to(device)
    model.configure(optimizer=torch.optim.Adam(model.parameters(),lr=args.lr),loss=None,metrics=model.metrics,logdir=args.logdir)

    model.fit(train_loader,dev=dev_loader,epochs=args.epochs)

    # Generate predictions manually, passing only inputs to predict_step
    preds = []
    # for batch in dev_loader:
    for batch in test_loader:
        # Unpack full batch: inputs and targets
        (xs_padded, x_lengths), (ys_padded, y_lengths) = batch
        # Move only inputs to device (targets not used in predict_step)
        xs_padded = xs_padded.to(device)
        x_lengths = x_lengths.to(device)
        # Reconstruct batch tuple for predict_step
        batch_device = ((xs_padded, x_lengths), (ys_padded, y_lengths))
        batch_preds = model.predict_step(batch_device, as_numpy=True)
        preds.extend(batch_preds)
    os.makedirs(args.logdir,exist_ok=True)
    with open(os.path.join(args.logdir,"speech_recognition_dev.txt"),"w",encoding="utf-8") as f:
        for seq in preds:
            f.write("".join(CommonVoiceCs.LETTER_NAMES[i] for i in seq.tolist())+"\n")

if __name__=="__main__":
    args=parser.parse_args()
    main(args)
