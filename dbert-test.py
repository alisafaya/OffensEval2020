# this source code was adapted from the source in this link: https://mccormickml.com/2019/07/22/BERT-fine-tuning/
import numpy as np
import time
import datetime
import torch
import random
import json
import os
import sys
from sklearn.metrics import classification_report, confusion_matrix
from transformers import *
from nltk.tokenize import sent_tokenize
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

# Tell pytorch to run this model on the GPU.
set_id = sys.argv[1]
use_gpu = True
seed = 1234
batch_size = 24
max_length = 256
label_list = [0, 1]
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-multilingual-cased')
model_path = "OffensEval/"+ set_id +"_model"

# If there's a GPU available...
if use_gpu and torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

def format_time(elapsed):
    return str(datetime.timedelta(seconds=int(round((elapsed)))))

def prepare_set(dataset, max_length=256):
    """returns input_ids, input_masks, labels for set of data ready in BERT format"""
    global tokenizer
    input_ids, input_masks, labels = [], [], []
    
    for i in dataset:
        input_ids.append(i["text"])
        labels.append(1 if i["label"] == 1 else 0)
    
    input_ids = [ tokenizer.encode(i, pad_to_max_length=True, add_special_tokens=True, max_length=max_length) for i in input_ids ]

    for sent in input_ids:
        att_mask = [int(token_id > 0) for token_id in sent]
        input_masks.append(att_mask)

    input_ids = torch.tensor(input_ids)
    input_masks = torch.tensor(input_masks)
    labels = torch.tensor(labels)

    return input_ids, input_masks, labels

def predict_sample(model, sample):
    input_ids, input_masks, labels = tuple(t.to(device) for t in prepare_set([sample,]))
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=input_masks) 
    logits = outputs[0].detach().cpu().numpy()
    return logits

def calculate_proba(results):
    ps = np.exp(results) 
    ps /= np.reshape(np.repeat(np.sum(ps, axis=1), 2), (ps.shape[0], 2)) 
    return np.argmax(np.sum(ps, axis=0))

def calculate_argmax(results):
    return np.argmax(results, axis=1).flatten()

if __name__ == "__main__":
    all_true = []
    all_pred = []
    for fold in range(1, 5):
        model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-multilingual-cased', num_labels=2)
        model.load_state_dict(torch.load(os.path.join(model_path, "model_%d_.pt" % (fold,))))
        model.to(device)
        with open(os.path.join(model_path, "fold_%d.json" % (fold,)), "r") as fo: 
            current_fold = json.loads(fo.read())
        for s in current_fold:
            results = predict_sample(model, s) 
            s["seq_pred"] = int(calculate_argmax(results))

        y_true = [ 1 if u["label"] == 1 else 0 for u in current_fold ]
        y_pred = [ u["seq_pred"] for u in current_fold ]
        all_true += y_true
        all_pred += y_pred

        print(classification_report(y_true, y_pred))
        
        with open(os.path.join(model_path, "pfold_%d.json" % (fold,)), "w") as fo:
            fo.write(json.dumps(list(current_fold), ensure_ascii=False))
        print("finished fold no", fold)
    print("Total evaluation:\n", classification_report(all_true, all_pred))
    print(confusion_matrix(all_true, all_pred))
