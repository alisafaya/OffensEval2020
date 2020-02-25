from data import read_file, fold_iterator
from models import *
import numpy as np
import time
import datetime
import torch
import random
import json
import os
import sys
from transformers import *
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from sklearn.metrics import classification_report, confusion_matrix
import unicodedata

set_id = sys.argv[1]

if set_id == "tr":
    pretrained_model = 'dbmdz/bert-base-turkish-cased'
elif set_id == "greek":
    pretrained_model = 'nlpaueb/bert-base-greek-uncased-v1'
elif set_id == "da":
    pretrained_model = './OffensEval/danish_bert_uncased_v2/'

output_id = sys.argv[0].split('.')[0]
use_gpu = True
seed = 1234
batch_size = 64
max_length = 64
label_list = [0, 1]
folds = 4
n_epochs = 10
lr = 1e-5

tokenizer = AutoTokenizer.from_pretrained(pretrained_model) if set_id != "da" else BertTokenizer.from_pretrained(pretrained_model)
model_path = "OffensEval/"+ set_id +"_berturk_cnn_model"
output_path = './output/' + set_id + '/' + output_id

if use_gpu and torch.cuda.is_available():
    device = torch.device("cuda:5")
else:
    device = torch.device("cpu")

def strip_accents_and_lowercase(s):
   return ''.join(c for c in unicodedata.normalize('NFD', s)
                  if unicodedata.category(c) != 'Mn').lower()

def prepare_set(dataset, max_length=256):
    """returns input_ids, input_masks, labels for set of data ready in BERT format"""
    global tokenizer
    input_ids, input_masks, labels = [], [], []
    
    for i in dataset:
        input_ids.append(i["text"].lower() if set_id != "greek" else strip_accents_and_lowercase(i["text"]))
        labels.append(1 if i["label"] == 1 else 0)
    
    input_ids = [ tokenizer.encode(i, pad_to_max_length=True, add_special_tokens=True, max_length=max_length) for i in input_ids ]

    for sent in input_ids:
        att_mask = [int(token_id > 0) for token_id in sent]
        input_masks.append(att_mask)

    input_ids = torch.tensor(input_ids).to(device)
    input_masks = torch.tensor(input_masks).to(device)
    labels = torch.FloatTensor(labels).unsqueeze(1).to(device)

    return input_ids, input_masks, labels

def generate_batch_data(x, y, batch_size):
    i, batch = 0, 0
    for batch, i in enumerate(range(0, len(x) - batch_size, batch_size), 1):
        x_batch = x[i : i + batch_size]
        y_batch = y[i : i + batch_size]
        yield x_batch, y_batch, batch
    if i + batch_size < len(x):
        yield x[i + batch_size :], y[i + batch_size :], batch + 1
    if batch == 0:
        yield x, y, 1


class AttentionNetBert(nn.Module):
    def __init__(self, embed_size, maxlen, bert_model):
        super(AttentionNetBert, self).__init__()
        drp = 0.1
        self.embedding_dropout = nn.Dropout2d(0.1)
        self.lstm = nn.LSTM(embed_size, 128, bidirectional=True, batch_first=True)
        self.lstm2 = nn.GRU(128*2, 64, bidirectional=True, batch_first=True)
        self.attention_layer = Attention(128, maxlen)
        self.linear = nn.Linear(64*2 , 64)
        self.gelu = nn.GELU()
        self.out = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()
        self.bert_model = bert_model

    def forward(self, x):
        h_embedding = self.bert_model(x)[0]
        h_embedding = self.embedding_dropout(h_embedding)
        h_lstm, _ = self.lstm(h_embedding)
        h_lstm, _ = self.lstm2(h_lstm)
        h_lstm_atten = self.attention_layer(h_lstm)
        conc = self.gelu(self.linear(h_lstm_atten))
        out = self.out(conc)
        return self.sigmoid(out)


if __name__ == "__main__":
    fold_no = 0
    all_data = read_file(set_id)
    all_pred, all_test, all_probs = [], [], []
    for train, dev, test in fold_iterator_sklearn(all_data, K=folds, random_seed=seed):
    
        if set_id == "da":
            config = AutoConfig.from_pretrained(pretrained_model + 'config.json')
            bert_model = AutoModel.from_pretrained(pretrained_model, config=config)
        else:
            bert_model = AutoModel.from_pretrained(pretrained_model)
    
    # ###
    # random.seed(seed)
    # random.shuffle(all_data) # initial shuffle
    # all_data = np.array(all_data) # convert to numpy for list indexing
    # dev_size = int(len(all_data) * 0.15)
    # model_path = "final"
    # for train, dev in [(all_data[dev_size:], all_data[:dev_size]), ]:
    # ###

        fold_no += 1
        print("Starting training fold number", fold_no)
        print([len(x) for x in (train, dev, test)])
        train_inputs, train_masks, y_train = prepare_set(train, max_length=max_length)
        dev_inputs, dev_masks, y_val = prepare_set(dev, max_length=max_length)
        test_inputs, test_masks, y_test = prepare_set(test, max_length=max_length)

        model = AttentionNetBert(768, max_length, bert_model)
        model.to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        loss_fn = nn.BCELoss()
        train_losses, val_losses = [], []
        
        best_dev = 1e20
        for epoch in range(n_epochs):
            start_time = time.time()
            train_loss = 0 
            model.train(True)

            for x_batch, y_batch, batch in generate_batch_data(train_inputs, y_train, batch_size):
                optimizer.zero_grad()
                y_pred = model(x_batch)
                loss = loss_fn(y_pred, y_batch)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            train_losses.append(train_loss)
            elapsed = time.time() - start_time
            model.eval()

            with torch.no_grad(): 

                val_loss, batch = 0, 1
                for x_batch, y_batch, batch in generate_batch_data(dev_inputs, y_val, batch_size):
                    y_pred = model(x_batch)
                    loss = loss_fn(y_pred, y_batch)
                    val_loss += loss.item()

            val_losses.append(val_loss)    
            print("Epoch %d Train loss: %.4f. Validation loss: %.4f. Elapsed time: %.2fs."% (epoch + 1, train_losses[-1], val_losses[-1], elapsed))

        #     if best_dev > val_loss:
        #         torch.save(model.state_dict(), os.path.join(model_path, "model_%d_.pt" % (fold_no,)))
        #         best_dev = val_loss

        # model.load_state_dict(torch.load(os.path.join(model_path, "model_%d_.pt" % (fold_no,))))
        # model.to(device)
        all_test += y_test.cpu().numpy().tolist()
        model.eval()
        with torch.no_grad():
            y_preds = []
            print("Evaluating fold", fold_no)
            for x_batch, y_batch, batch in generate_batch_data(test_inputs, y_test, batch_size):
                y_pred = model(x_batch)
                y_pred = y_pred.cpu().numpy().flatten()
                all_probs += list(y_pred)
                y_preds += [ 1 if p >= 0.5 else 0 for p in y_pred ] 
            
            print(classification_report(y_test.cpu().numpy().tolist(), y_preds))
        all_pred += y_preds

        del model
        torch.cuda.empty_cache()

    print("Finished", fold_no, "Evaluation")
    print(classification_report(all_test, all_pred))
    np.save(output_path + ".probs", np.array(all_probs))
    np.save(output_path + ".gold", np.array(all_test))


