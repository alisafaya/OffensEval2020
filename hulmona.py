from fastai import *
from fastai.text import *
import string
from sklearn.model_selection import train_test_split
from data import *
import pandas as pd

seed = 1234
folds = 4
path = Path('.')

arabic_punctuations = '''`÷×؛<>_()*&^%][ـ،/:"؟.,'{}~¦+|!”…“–ـ«»'''
english_punctuations = string.punctuation
punc_to_remove = ''.join(set(arabic_punctuations + english_punctuations))
punc_to_keep = '+'
punc_to_escape = '''[]-^'''
for p in punc_to_keep: punc_to_remove = punc_to_remove.replace(p, '')
for p in punc_to_escape: punc_to_remove = punc_to_remove.replace(p, '\\{}'.format(p))

def pre_process(text):
    text = text.replace('\\n', ' ').replace('\n', ' ')
    text = text.replace('؛', '،')
    text = re.sub(r'\([^)]+\)', '', text)  # remove parentheses and everything in between
    text = re.sub(r'[a-zA-Z]', '', text)  # remove non-arabic characters
    text = re.sub(r'\d+(\.\d+)?', ' رقم ', text)  # replace numbers by special token
    for p in punc_to_remove: text = text.replace(p, '')  # remove punctuations
    text = re.sub(r'(.)\1{2,}', r'\1', text)  # remove repeated chars
    text = re.sub(r'\s+', r' ', text)
    return text

pretrained_fnames=['Ar-LM-epoch1-acc43','itos']

all_data = read_file("ar")
train, dev, test = next(fold_iterator(all_data, K=folds, random_seed=seed))
ar_tok = Tokenizer(lang='ar')
df_train = pd.DataFrame(train.tolist())
df_val = pd.DataFrame(dev.tolist())

data_clas = TextClasDataBunch.from_df(path, train_df=df_train, valid_df=df_val, text_cols=1, label_cols=2, tokenizer=ar_tok, bs=32, include_bos=False, min_freq=4)
learn_clas = text_classifier_learner(data_clas, AWD_LSTM, drop_mult=0.2, metrics=[accuracy,FBeta(average='macro')])