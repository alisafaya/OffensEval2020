# adapted from https://mlwhiz.com/blog/2019/03/09/deeplearning_architectures_text_classification/
from torch import nn
import torch.nn.functional as F
import torch


class CNN_Text(nn.Module):
    def __init__(self, embed_size, max_features):
        super(CNN_Text, self).__init__()
        
        filter_sizes = [1,2,3,5]
        num_filters = 36

        self.embedding = nn.Embedding(max_features, embed_size)
        self.convs1 = nn.ModuleList([nn.Conv2d(1, num_filters, (K, embed_size)) for K in filter_sizes])
        self.dropout = nn.Dropout(0.1)
        self.fc1 = nn.Linear(len(filter_sizes) * num_filters, 1)

    def forward(self, x):
        x = self.embedding(x)  
        x = x.unsqueeze(1)  
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs1] 
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  
        x = torch.cat(x, 1)
        x = self.dropout(x)  
        logit = self.fc1(x)  
        return logit


class BiLSTM(nn.Module):
    
    def __init__(self, embed_size, max_features):
        super(BiLSTM, self).__init__()
        drp = 0.1
        self.hidden_size = 128
        self.embedding = nn.Embedding(max_features, embed_size)
        self.lstm = nn.LSTM(embed_size, self.hidden_size, bidirectional=True, batch_first=True)
        self.linear = nn.Linear(self.hidden_size*4 , 128)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(drp)
        self.out = nn.Linear(128, 1)


    def forward(self, x):
        h_embedding = self.embedding(x)
        h_embedding = torch.squeeze(torch.unsqueeze(h_embedding, 0))
        h_lstm, _ = self.lstm(h_embedding)
        avg_pool = torch.mean(h_lstm, 1)
        max_pool, _ = torch.max(h_lstm, 1)
        conc = torch.cat(( avg_pool, max_pool), 1)
        conc = self.relu(self.linear(conc))
        conc = self.dropout(conc)
        out = self.out(conc)
        return out


class CNNBert(nn.Module):
    
    def __init__(self, embed_size, bert_model):
        super(CNNBert, self).__init__()
        filter_sizes = [1,2,3,4,5]
        num_filters = 32
        self.convs1 = nn.ModuleList([nn.Conv2d(4, num_filters, (K, embed_size)) for K in filter_sizes])
        self.dropout = nn.Dropout(0.1)
        self.fc1 = nn.Linear(len(filter_sizes)*num_filters, 1)
        self.sigmoid = nn.Sigmoid()
        self.bert_model = bert_model

    def forward(self, x, input_masks, token_type_ids):
        x = self.bert_model(x, attention_mask=input_masks, token_type_ids=token_type_ids)[2][-4:]
        x = torch.stack(x, dim=1)
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs1] 
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  
        x = torch.cat(x, 1)
        x = self.dropout(x)  
        logit = self.fc1(x)
        return self.sigmoid(logit)
