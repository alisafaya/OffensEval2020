from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer
from torch.optim import Adam
from data import *
from models import *
import torch
import numpy as np
import torch.nn as nn
import unicodedata
import json 
import time 
import re
import os

def initialize_tokenizer(_set):
    global tokenizer

    if _set == "tr":
        tokenizer_id = 'dbmdz/bert-base-turkish-cased'
    elif _set == "gr":
        tokenizer_id = 'nlpaueb/bert-base-greek-uncased-v1'
    elif _set == "ar":
        tokenizer_id = 'asafaya/bert-base-arabic'
    tokenizer = BertTokenizer.from_pretrained(tokenizer_id)

def preprocess_text(identifier): 
    # https://stackoverflow.com/a/29920015/5909675
    matches = re.finditer('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)', identifier.replace("#", " "))
    return " ".join([m.group(0) for m in matches])


def strip_accents_and_lowercase(s):
   return ''.join(c for c in unicodedata.normalize('NFD', s)
                  if unicodedata.category(c) != 'Mn').lower()


def prepare_set(dataset, _set, max_length=64):
    """returns input_ids, input_masks, labels for set of data ready in BERT format"""
    global tokenizer
    input_ids, labels = [], []
    for i in dataset:
        input_ids.append(preprocess_text(i["text"]) if _set != "gr" else strip_accents_and_lowercase(preprocess_text(i["text"])))
        labels.append(1 if i["label"] == 1 else 0)

    tokenized = tokenizer.batch_encode_plus(input_ids, pad_to_max_length=True, add_special_tokens=True, max_length=max_length, return_tensors="pt")["input_ids"]
    labels = torch.FloatTensor(labels).unsqueeze(1)

    return tokenized, labels


def train(_set, model):
    train_samples = read_file(_set +".train")
    x, y = prepare_set(train_samples, _set, max_length=max_length)
    dev_size = int(len(x) * 0.10)
    x_train, x_dev, y_train, y_dev = x[dev_size:], x[:dev_size], y[dev_size:], y[:dev_size]

    # Create the DataLoader for training set.
    train_data = TensorDataset(x_train, y_train)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

    # Create the DataLoader for dev set.
    dev_data = TensorDataset(x_dev, y_dev)
    dev_sampler = SequentialSampler(dev_data)
    dev_dataloader = DataLoader(dev_data, sampler=dev_sampler, batch_size=batch_size)

    model.to(device)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    optimizer = Adam(model.parameters(), lr=lr)
    criterion = torch.nn.BCEWithLogitsLoss()
    model.zero_grad()
    best_score = 0
    best_loss = 1e6
    for epoch in range(n_epochs):

        start_time = time.time()
        train_loss = 0 
        model.train()

        for batch in train_dataloader:
            b_input_ids, b_labels = tuple(t.to(device) for t in batch)
            output = model(b_input_ids)
            loss = criterion(output, b_labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
            model.zero_grad()

        train_loss /= len(train_dataloader)  

        elapsed = time.time() - start_time
        model.eval()
        val_preds = []
        with torch.no_grad():
            val_loss = 0
            for batch in dev_dataloader:
                b_input_ids, b_labels = tuple(t.to(device) for t in batch)
                output = model(b_input_ids)
                loss = criterion(output, b_labels)
                val_loss += loss.item()
                preds = torch.sigmoid(output).detach().cpu().numpy().flatten()
                val_preds += list(preds)
                model.zero_grad()

        val_loss /= len(dev_dataloader)
        val_preds = [ int(x >= 0.5) for x in val_preds ]
        val_score = f1_score(y_dev, val_preds, average="macro")
        # print("Epoch %d Train loss: %.4f. Validation F1-Score: %.4f  Validation loss: %.4f. Elapsed time: %.2fs."% (epoch + 1, train_loss, val_score, val_loss, elapsed))

        if val_score > best_score:
            torch.save(model.state_dict(), model_path)
            # print(classification_report(y_dev, val_preds, digits=3))
            best_score = val_score

    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.predict = predict.__get__(model)
    os.remove(model_path)
    return model


def predict(model, x):
    # Create the DataLoader for dev set.
    
    data = TensorDataset(x)
    sampler = SequentialSampler(data)
    dataloader = DataLoader(data, sampler=sampler, batch_size=batch_size)
    model.eval()
    preds = []
    with torch.no_grad():
        for b_input_ids in dataloader:
            b_input_ids = b_input_ids[0].to(device)
            output = model(b_input_ids)
            probs = torch.sigmoid(output).detach().cpu().numpy().flatten()
            preds += list(probs)
            model.zero_grad()

    return [ int(x >= 0.5) for x in preds ]


def evaluate(_set, M):
    # Preprocessing
    print("Training", M,"for:", _set)
    initialize_tokenizer(_set)

    model = M(embed_size, tokenizer.vocab_size)
    model = train(_set, model)

    # Testing
    print("Testing", M , "for:", _set)
    test_samples = read_file(_set +".test")
    x, _ = prepare_set(test_samples, _set, max_length=max_length)
    y_test = [ x["label"] for x in test_samples ]
    predictions = model.predict(x)
    print ('Test data\n', classification_report(y_test, predictions, digits=3))

    return

max_length = 64
tokenizer = None
batch_size = 32
seed = 1234
n_epochs = 10
embed_size = 300
lr = 0.001
model_path = "temp.pt"
use_gpu = True

if use_gpu and torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

if __name__ == "__main__":
    for _set in ("ar", "gr", "tr"):
        for m in (CNN_Text, BiLSTM):
            evaluate(_set, m)