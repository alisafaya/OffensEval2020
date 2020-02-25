# this source code was adapted from the source in this link: https://mccormickml.com/2019/07/22/BERT-fine-tuning/
import numpy as np
import time
import datetime
import torch
import random
import json
import os
import sys
from transformers import *
import data
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

# Tell pytorch to run this model on the GPU.
set_id = sys.argv[1]
use_gpu = True
seed = 1234
batch_size = 8
max_length = 256
label_list = [0, 1]
folds = 4
epochs = 6
model_path = "OffensEval/"+ set_id +"_model"

if set_id == "tr":
    pretrained_model = 'dbmdz/bert-base-turkish-cased'
elif set_id == "greek":
    pretrained_model = 'nlpaueb/bert-base-greek-uncased-v1'
elif set_id == "da":
    pretrained_model = './OffensEval/danish_bert_uncased/'

tokenizer = BertTokenizer.from_pretrained(pretrained_model)

# If there's a GPU available...
if use_gpu and torch.cuda.is_available():
    device = torch.device("cuda:1")
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


if __name__ == "__main__":
    fold_no = 0
    all_data = data.read_file(set_id)

    for train, dev, test in data.fold_iterator(all_data, K=folds, random_seed=seed):
    
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
        if set_id == "da":
            config = BertConfig.from_pretrained(pretrained_model + 'config.json')
            model = BertForSequenceClassification.from_pretrained(pretrained_model, config=config)
        else:
            model = BertForSequenceClassification.from_pretrained(pretrained_model)

        with open(os.path.join(model_path, "fold_%d.json" % (fold_no,)), "w") as fo:
            fo.write(json.dumps(list(test), ensure_ascii=False))
        if device.type == "cuda":
            model.to(device)

        train_inputs, train_masks, train_labels = prepare_set(train, max_length=max_length)
        dev_inputs, dev_masks, dev_labels = prepare_set(dev, max_length=max_length)

        # Create the DataLoader for our training set.
        train_data = TensorDataset(train_inputs, train_masks, train_labels)
        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

        # Create the DataLoader for our dev set.
        dev_data = TensorDataset(dev_inputs, dev_masks, dev_labels)
        dev_sampler = SequentialSampler(dev_data)
        dev_dataloader = DataLoader(dev_data, sampler=dev_sampler, batch_size=batch_size)

        optimizer = AdamW(model.parameters(), lr = 5e-6)

        # Total number of training steps is number of batches * number of epochs.
        total_steps = len(train_dataloader) * epochs

        np.random.seed(seed)
        torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed)

        # Create the learning rate scheduler.
        scheduler = get_linear_schedule_with_warmup(optimizer, 
                                            num_warmup_steps = 0, # Default value in run_glue.py
                                            num_training_steps = total_steps)
        loss_values = []
        model.zero_grad()
        last_dev = 1e20
        best_dev = 1e20
        
        # For each epoch...
        for epoch_i in range(0, epochs):
            
            # Perform one full pass over the training set.
            print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
            print('Training...')

            # Measure how long the training epoch takes.
            t0 = time.time()
            total_loss = 0
                
            # For each batch of training data...
            for step, batch in enumerate(train_dataloader):

                model.train()
                # Unpack this training batch from our dataloader. 
                # As we unpack the batch, we'll also copy each tensor to the GPU using the 
                # `to` method.
                if device.type == "cuda":
                    batch = tuple(t.to(device) for t in batch)
                b_input_ids, b_input_mask, b_labels = batch

                # Forward pass (evaluate the model on this training batch)
                outputs = model(b_input_ids, 
                            attention_mask=b_input_mask, 
                            labels=b_labels)
                
                loss = outputs[0]

                # Accumulate the loss. `loss` is a Tensor containing a single value; 
                # the `.item()` function just returns the Python value from the tensor.
                total_loss += loss.item()
                # Perform a backward pass to calculate the gradients.
                loss.backward()
                # Clip the norm of the gradients to 1.0.
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                # Update parameters and take a step using the computed gradient
                optimizer.step()
                # Update lr and Clear out the gradients (by default they accumulate)
                scheduler.step()
                model.zero_grad()

            # Calculate the average loss over the training data.
            avg_train_loss = total_loss / len(train_dataloader)            
            loss_values.append(avg_train_loss)

            print("Average training loss: {0:.2f}".format(avg_train_loss))
            print("Training epoch took: {:}".format(format_time(time.time() - t0)))
                
            # ========================================
            #               Validation
            # ========================================
            # After the completion of each training epoch, measure our performance on
            # our validation set.

            t0 = time.time()

            # Tracking variables 
            dev_loss, eval_accuracy = 0, 0
            nb_eval_steps, nb_eval_examples = 0, 0
            model.eval() 
            # Evaluate data for one epoch
            for batch in dev_dataloader:
                
                # Add batch to GPU
                if device.type == "cuda":
                    batch = tuple(t.to(device) for t in batch)
                
                # Unpack the inputs from our dataloader
                b_input_ids, b_input_mask, b_labels = batch
                
                outputs = model(b_input_ids, 
                            attention_mask=b_input_mask, 
                            labels=b_labels)
                
                loss = outputs[0]

                # Accumulate the loss. `loss` is a Tensor containing a single value; 
                # the `.item()` function just returns the Python value from the tensor.
                dev_loss += loss.item()

                # Track the number of batches
                nb_eval_steps += 1

                # Clear out the gradients (by default they accumulate)
                model.zero_grad()

            # Report the final accuracy for this validation run.
            print("Validation loss: {0:.2f}".format(dev_loss/nb_eval_steps))
            print("Validation took: {:}".format(format_time(time.time() - t0)))
            
            # Keep track of the best development accuracy, and save the model only if it's the best one
            if best_dev > dev_loss:
                torch.save(model.state_dict(), os.path.join(model_path, "model_%d_.pt" % (fold_no,)))
                best_dev = dev_loss

        print("Training complete!")


