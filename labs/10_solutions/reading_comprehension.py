#!/usr/bin/env python3
# Group IDs: 31ff17c9-b0b8-449e-b0ef-8a1aa1e14eb3, 5b78caaa-8040-46f7-bf54-c13e183bbbf8

import argparse
import random
import numpy as np
import os
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline

import npfl138
from npfl138 import TrainableModule
from npfl138.datasets.reading_comprehension_dataset import ReadingComprehensionDataset

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=8, help="Batch size.")
parser.add_argument('--learning_rate', type=float, default=3e-5, help="Learning rate.")
parser.add_argument('--max_length', type=int, default=512, help="Max length of input sequences.")
parser.add_argument('--epochs', type=int, default=3, help="Number of epochs.")
parser.add_argument('--doc_stride', type=int, default=128, help="Document stride for tokenization.")
parser.add_argument('--seed', type=int, default=42, help="Random seed.")
parser.add_argument('--threads', type=int, default=4, help="Maximum number of threads to use.")


class QATrainDataset(Dataset):
    """Dataset for QA, providing tokenized inputs and target answer spans."""
    def __init__(self, dataset, tokenizer, max_length, doc_stride):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.doc_stride = doc_stride
        self.samples = []
        for pi, paragraph in enumerate(self.dataset.paragraphs):
            for qi, _ in enumerate(paragraph["qas"]):
                self.samples.append((pi, qi))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        pi, qi = self.samples[idx]
        para = self.dataset.paragraphs[pi]
        context = para['context']
        qa = para['qas'][qi]
        question = qa['question']
        answer = qa['answers'][0]
        ans_text, ans_start = answer['text'], answer['start']

        encoding = self.tokenizer(
            question,
            context,
            truncation='only_second',
            max_length=self.max_length,
            stride=self.doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True
        )
        ans_end = ans_start + len(ans_text)
        for i in range(len(encoding['input_ids'])):
            seq_ids = encoding.sequence_ids(i)
            c_start = next((j for j, s in enumerate(seq_ids) if s == 1), None)
            c_end = len(seq_ids) - 1 - next((j for j, s in enumerate(reversed(seq_ids)) if s == 1), 0)
            if c_start is None or c_end is None:
                continue
            offsets = encoding['offset_mapping'][i]
            if offsets[c_start][0] <= ans_start and ans_end <= offsets[c_end][1]:
                start_idx = next(j for j in range(c_start, c_end + 1)
                                 if offsets[j][0] <= ans_start < offsets[j][1])
                end_idx = next(j for j in range(c_start, c_end + 1)
                               if offsets[j][0] < ans_end <= offsets[j][1])
                sample = {
                    'input_ids': encoding['input_ids'][i],
                    'attention_mask': encoding['attention_mask'][i],
                    'start_idx': start_idx,
                    'end_idx': end_idx
                }
                if 'token_type_ids' in encoding:
                    sample['token_type_ids'] = encoding['token_type_ids'][i]
                return sample
        raise ValueError(f"Answer not in any chunk for sample {idx}")

    def collate(self, batch):
        max_len = max(len(s['input_ids']) for s in batch)
        pad_id = self.tokenizer.pad_token_id or 0
        input_ids, attention_mask, token_type_ids = [], [], []
        starts, ends = [], []
        for s in batch:
            seq, mask = s['input_ids'], s['attention_mask']
            tt = s.get('token_type_ids')
            pad_len = max_len - len(seq)
            seq = seq + [pad_id] * pad_len
            mask = mask + [0] * pad_len
            if tt is not None:
                tt = tt + [0] * pad_len
            input_ids.append(seq)
            attention_mask.append(mask)
            token_type_ids.append(tt if tt is not None else [0] * max_len)
            starts.append(s['start_idx'])
            ends.append(s['end_idx'])
        input_ids = torch.tensor(input_ids)
        attention_mask = torch.tensor(attention_mask)
        token_type_ids = torch.tensor(token_type_ids) if any('token_type_ids' in s for s in batch) else None
        starts = torch.tensor(starts)
        ends = torch.tensor(ends)
        inputs = (input_ids, attention_mask, token_type_ids) if token_type_ids is not None else (input_ids, attention_mask)
        return inputs, (starts, ends)

class QAModel(TrainableModule):
    def __init__(self, model_name, max_length, doc_stride):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForQuestionAnswering.from_pretrained(model_name)
        self.max_length = max_length
        self.doc_stride = doc_stride

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        out = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        ) if token_type_ids is not None else self.model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        return out.start_logits, out.end_logits

    def compute_loss(self, preds, labels, *args, **kwargs):
        start_logits, end_logits = preds
        start_labels, end_labels = labels
        loss_s = F.cross_entropy(start_logits, start_labels)
        loss_e = F.cross_entropy(end_logits, end_labels)
        return (loss_s + loss_e) / 2

    def predict_answers(self, paragraphs):
        self.eval()
        device = self.device or torch.device('cpu')
        preds = []
        for para in paragraphs:
            context = para['context']
            for qa in para['qas']:
                question = qa['question']
                enc = self.tokenizer(
                    question,
                    context,
                    truncation='only_second',
                    max_length=self.max_length,
                    stride=self.doc_stride,
                    return_overflowing_tokens=True,
                    return_offsets_mapping=True
                )
                best_text, best_score = '', -1e9 # best for now
                for i in range(len(enc['input_ids'])):
                    inputs = {
                        'input_ids': torch.tensor([enc['input_ids'][i]], device=device),
                        'attention_mask': torch.tensor([enc['attention_mask'][i]], device=device)
                    }
                    if 'token_type_ids' in enc:
                        inputs['token_type_ids'] = torch.tensor([enc['token_type_ids'][i]], device=device)
                    with torch.no_grad():
                        out = self.model(**inputs)
                    start_logits = out.start_logits[0].cpu().numpy()
                    end_logits = out.end_logits[0].cpu().numpy()
                    offsets = enc['offset_mapping'][i]
                    seq_ids = enc.sequence_ids(i)
                    c_start = next((j for j, s in enumerate(seq_ids) if s == 1), None)
                    c_end = len(seq_ids) - 1 - next((j for j, s in enumerate(reversed(seq_ids)) if s == 1), 0)
                    if c_start is None or c_end is None:
                        continue
                    # precompute best end for each start
                    max_end = -1e9
                    best_end = c_end
                    end_from = [0] * (c_end + 1)
                    end_idx_from = [c_end] * (c_end + 1)
                    for j in range(c_end, c_start - 1, -1):
                        if end_logits[j] > max_end:
                            max_end = end_logits[j]
                            best_end = j
                        end_from[j] = max_end
                        end_idx_from[j] = best_end
                    for j in range(c_start, c_end + 1):
                        score = start_logits[j] + end_from[j]
                        if score > best_score:
                            best_score = score
                            st, en = j, end_idx_from[j]
                            if offsets[st] and offsets[en]:
                                best_text = context[offsets[st][0]:offsets[en][1]]
                preds.append(best_text)
        return preds

def main(args: argparse.Namespace) -> None:
    npfl138.startup(args.seed, args.threads)
    npfl138.global_keras_initializers()

    # Load data and model
    ds = ReadingComprehensionDataset()
    model = QAModel('ufal/robeczech-base', args.max_length, args.doc_stride)

    # Prepare train and dev loaders
    train_ds = QATrainDataset(ds.train, model.tokenizer, args.max_length, args.doc_stride)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=train_ds.collate
    )
    dev_ds = QATrainDataset(ds.dev, model.tokenizer, args.max_length, args.doc_stride)
    dev_loader = DataLoader(
        dev_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=dev_ds.collate
    )

    # Dev accuracy callback
    class DevAcc:
        def __init__(self, dev_ds):
            self.dev = dev_ds
        def __call__(self, mdl, epoch, logs):
            preds = mdl.predict_answers(self.dev.paragraphs)
            acc = ReadingComprehensionDataset.evaluate(self.dev, preds)
            logs['dev_accuracy'] = acc

    dev_callback = DevAcc(ds.dev)

    # Configure & train
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    model.configure(optimizer=optimizer, device='auto')
    model.fit(
        train_loader,
        epochs=args.epochs,
        dev=dev_loader,
        callbacks=[dev_callback],
        console=2
    )

    # Inference with trained model
    qa_pipe = pipeline(
        'question-answering',
        model=model.model,
        tokenizer=model.tokenizer,
        device=0 if torch.cuda.is_available() else -1
    )
    inputs = [
        {'question': qa['question'], 'context': para['context']}
        for para in ds.test.paragraphs for qa in para['qas']
    ]
    results = qa_pipe(inputs, batch_size=args.batch_size)
    test_preds = [res['answer'] for res in results]

    # Save predictions
    os.makedirs(args.logdir, exist_ok=True)
    out_path = os.path.join(args.logdir, 'reading_comprehension.txt')
    with open(out_path, 'w', encoding='utf-8') as predictions_file:
        for answer in test_preds:
            print(answer, file=predictions_file)
    print(f"Predictions saved to {out_path}")

if __name__ == '__main__':
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    main(main_args)