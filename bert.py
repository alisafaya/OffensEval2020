from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from transformers import *
from torch import nn
from data import *
import numpy as np
import time
import datetime
import torch
import random
import json
import os
import sys
import torch.nn.functional as F
import unicodedata
import re

set_id = sys.argv[1]
use_gpu = True
seed = 1234
device_ids = [0, 1, 2, 3, 4, 5, 6, 7]
batch_size = 32 * len(device_ids)
max_length = 64
lr = 2e-5

if use_gpu and torch.cuda.is_available():
    device = torch.device("cuda:%d"%(device_ids[0]))
else:
    device = torch.device("cpu")

tokenizer = None

# if set_id == "tr":
#     pretrained_model = 'dbmdz/bert-base-turkish-cased'
# elif set_id == "gr":
#     pretrained_model = 'nlpaueb/bert-base-greek-uncased-v1'
# elif set_id == "ar":
#     pretrained_model = 'asafaya/bert-base-arabic'

def preprocess_text(identifier): 
    # https://stackoverflow.com/a/29920015/5909675
    matches = re.finditer('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)', identifier.replace("#", " "))
    return " ".join([m.group(0) for m in matches])

def strip_accents_and_lowercase(s):
   return ''.join(c for c in unicodedata.normalize('NFD', s)
                  if unicodedata.category(c) != 'Mn').lower()

def prepare_set(text, max_length=64):
    """returns input_ids, attention_mask, token_type_ids for set of data ready in BERT format"""
    global tokenizer

    text = [ preprocess_text(t) if set_id != "gr" else strip_accents_and_lowercase(preprocess_text(t)) for t in text ]
    t = tokenizer.batch_encode_plus(text,
                        pad_to_max_length=True,
                        add_special_tokens=True,
                        max_length=max_length,
                        return_tensors='pt')

    return t["input_ids"], t["attention_mask"], t["token_type_ids"]

def predict(self, test_set, batch_size=batch_size):
    test_inputs, test_masks, test_type_ids = prepare_set(test_set)
    test_data = TensorDataset(test_inputs, test_masks, test_type_ids)
    test_sampler = SequentialSampler(test_data)
    test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=batch_size)

    self.eval()
    with torch.no_grad(): 
        preds = []
        for batch in test_dataloader:
            b_input_ids, b_input_mask, b_token_type_ids = tuple(t.to(device) for t in batch)
            output = self(b_input_ids, 
            attention_mask=b_input_mask,
            token_type_ids=b_token_type_ids)
            logits = output[0].detach().cpu()
            preds += list(torch.nn.functional.softmax(logits, dim=1)[:, 1].numpy())

    return preds

def train_bert(x_train, x_dev, y_train, y_dev, pretrained_model, n_epochs=10, model_path="temp.pt", batch_size=batch_size):
    global tokenizer
    tokenizer = BertTokenizer.from_pretrained(pretrained_model)
    model = BertForSequenceClassification.from_pretrained(pretrained_model)

    print([len(x) for x in (y_train, y_dev)])
    y_train, y_dev = ( torch.tensor(t) for t in (y_train, y_dev) )

    # Create the DataLoader for training set.
    train_inputs, train_masks, train_type_ids = prepare_set(x_train)
    train_data = TensorDataset(train_inputs, train_masks, train_type_ids, y_train)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

    # Create the DataLoader for dev set.
    dev_inputs, dev_masks, dev_type_ids = prepare_set(x_dev)
    dev_data = TensorDataset(dev_inputs, dev_masks, dev_type_ids, y_dev)
    dev_sampler = SequentialSampler(dev_data)
    dev_dataloader = DataLoader(dev_data, sampler=dev_sampler, batch_size=batch_size)

    if len(device_ids) > 1 and device.type == "cuda":
        model = nn.DataParallel(model, device_ids=device_ids)
    model.to(device)

    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    total_steps = len(train_dataloader) * n_epochs
    optimizer = AdamW(model.parameters(), lr=lr)
    scheduler = get_linear_schedule_with_warmup(optimizer, 
                                        num_warmup_steps = 0,
                                        num_training_steps = total_steps)

    model.zero_grad()
    best_score = 0
    best_loss = 1e6

    for epoch in range(n_epochs):

        start_time = time.time()
        train_loss = 0 
        model.train()

        for batch in train_dataloader:
            b_input_ids, b_input_mask, b_token_type_ids, b_labels = tuple(t.to(device) for t in batch)
            output = model(b_input_ids, 
                            attention_mask=b_input_mask,
                            token_type_ids=b_token_type_ids,
                            labels=b_labels)

            loss = output[0].sum()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
            model.zero_grad()

        train_loss /= len(train_dataloader)  

        scheduler.step()
        elapsed = time.time() - start_time
        model.eval()
        val_preds = []
        with torch.no_grad():
            val_loss, batch = 0, 1
            for batch in dev_dataloader:
                b_input_ids, b_input_mask, b_token_type_ids, b_labels = tuple(t.to(device) for t in batch)
                output = model(b_input_ids, 
                            attention_mask=b_input_mask,
                            token_type_ids=b_token_type_ids,
                            labels=b_labels)
                
                loss = output[0].sum()
                val_loss += loss.item()
                logits = output[1].detach().cpu().numpy()
                val_preds += list(np.argmax(logits, axis=1).flatten())
                model.zero_grad()

        val_loss /= len(dev_dataloader)
        val_score = f1_score(y_dev, val_preds, average="macro")
        print("Epoch %d Train loss: %.4f. Validation F1-Score: %.4f  Validation loss: %.4f. Elapsed time: %.2fs."% (epoch + 1, train_loss, val_score, val_loss, elapsed))

        if val_score > best_score:
            torch.save(model.state_dict(), model_path)
            print(classification_report(y_dev, val_preds, digits=4))
            best_score = val_score

    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.predict = predict.__get__(model)
    os.remove(model_path)

    return model


def evaluate():
    train_samples = read_file(set_id +".train")
    x, y = [ x["text"] for x in train_samples ], [ x["label"] for x in train_samples ]
    dev_size = int(len(x) * 0.10)
    x_train, x_dev, y_train, y_dev = x[dev_size:], x[:dev_size], y[dev_size:], y[:dev_size]
    model = train_bert(x_train, x_dev, y_train, y_dev, pretrained_model, n_epochs=5)

    # Testing
    test_samples = read_file(set_id +".test")
    x_test, y_test = [ x["text"] for x in test_samples ], [ x["label"] for x in test_samples ]
    predictions = model.predict(x_test)
    print ('Test data\n', classification_report(y_test, [ int(x >= 0.5) for x in predictions ], digits=3))

# pretrained_model = "bert-base-multilingual-uncased"

if __name__ == "__main__":
    evaluate()
    