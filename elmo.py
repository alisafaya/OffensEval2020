from data import *
from models import *
import numpy as np
import time
import datetime
import torch
import random
import json
import os
import sys
import logging
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from sklearn.metrics import classification_report, confusion_matrix
from nltk.tokenize import sent_tokenize, word_tokenize
from elmoformanylangs import Embedder
from keras.preprocessing.sequence import pad_sequences 

set_id = sys.argv[1]
output_id = sys.argv[0].split('.')[0]
logging.disable(logging.CRITICAL)
output_path = './output/' + set_id + '/' + output_id
use_gpu = True
seed = 1234
batch_size = 64
max_length = 64
label_list = [0, 1]
folds = 4
n_epochs = 15
lr = 5e-4
gpu_id = 1
model_path = "elmo_models/"+ set_id + "/"

if use_gpu and torch.cuda.is_available():
    device = torch.device("cuda:%d" % (gpu_id,))
    torch.cuda.set_device(gpu_id)
else:
    device = torch.device("cpu")

def prepare_set(dataset, max_length=256): 
    inputs, labels = [], [] 
    for i in dataset: 
        inputs.append(word_tokenize(i["text"])) 
        labels.append(1 if i["label"] == 1 else 0) 
    
    inputs = pad_sequences(inputs, dtype='object', truncating='post', maxlen=max_length, padding="pre", value=np.array("#", dtype="object")) 
    labels = torch.FloatTensor(labels).unsqueeze(1).to(device) 

    return inputs, labels 


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

class CNNBert(nn.Module):
    
    def __init__(self, embed_size, elmo_embedder):
        super(CNNBert, self).__init__()
        filter_sizes = [1,2,3,5]
        num_filters = 24
        self.convs1 = nn.ModuleList([nn.Conv2d(1, num_filters, (K, embed_size)) for K in filter_sizes])
        self.dropout = nn.Dropout(0.1)
        self.fc1 = nn.Linear(len(filter_sizes)*num_filters, 1)
        self.sigmoid = nn.Sigmoid()
        self.elmo_embedder = elmo_embedder

    def forward(self, x):
        x = torch.tensor(np.array(self.elmo_embedder.sents2elmo(x))).to(device)
        x = x.unsqueeze(1)
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs1] 
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  
        x = torch.cat(x, 1)
        x = self.dropout(x)  
        logit = self.fc1(x)
        return self.sigmoid(logit)


if __name__ == "__main__":
    fold_no = 0
    all_data = read_file(set_id)
    e = Embedder("elmo_models/%s/"%(set_id,))

    all_pred, all_test, all_probs = [], [], []
    for train, dev, test in fold_iterator_sklearn(all_data, K=folds, random_seed=seed):
    
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
        train_inputs, y_train = prepare_set(train, max_length=max_length)
        dev_inputs, y_val = prepare_set(dev, max_length=max_length)
        test_inputs, y_test = prepare_set(test, max_length=max_length)

        model = CNNBert(1024, e)
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
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
        # torch.cuda.empty_cache()

    print("Finished", fold_no, "Evaluation")
    print(classification_report(all_test, all_pred))
    np.save(output_path + ".probs", np.array(all_probs))
    np.save(output_path + ".gold", np.array(all_test))



