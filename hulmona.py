from fastai import *
from fastai.text import *
import string
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
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

df = pd.read_csv(path/'data'/'AR_FINAL.csv')
print(df.shape)
np.random.seed(42)
df_train, df_test = train_test_split(df, test_size=0.25)
df_train, df_val = train_test_split(df_train, test_size=0.1)
print(df_train.shape, df_val.shape)

ar_tok = Tokenizer(lang='ar')
data_lm = TextLMDataBunch.from_df(path, train_df=df_train, valid_df=df_val, text_cols=0, label_cols=None, tokenizer=ar_tok, bs=32)

learn_lm = language_model_learner(data_lm, AWD_LSTM, drop_mult=0.2, pretrained_fnames=pretrained_fnames)

data_clas = TextClasDataBunch.from_df(path, train_df=df_train, valid_df=df_val, text_cols=0, label_cols=1, tokenizer=ar_tok, bs=32, vocab=data_lm.train_ds.vocab, include_bos=False)
learn_clas = text_classifier_learner(data_clas, AWD_LSTM, drop_mult=0.2, metrics=[accuracy,FBeta(average='macro')])

learn_clas.fit_one_cycle(15, slice(5e-4/(2.6**4),5e-4))

test_results = [learn_clas.predict(t) for t in df_test["text"]]
test_results = [ r[1].item() for r in test_results ] 
print(classification_report(df_test["category"], test_results))