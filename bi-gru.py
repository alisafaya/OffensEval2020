from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import Dense, Embedding, CuDNNGRU, GRU, LSTM
from keras.utils import to_categorical
from keras.optimizers import Nadam
from keras.engine.input_layer import Input
from keras.models import Model
from keras_attention import AttentionWithContext
from keras.layers.wrappers import Bidirectional
from sklearn.metrics import classification_report, confusion_matrix
from data import *
import tensorflow as tf
from keras.optimizers import TFOptimizer
import sys
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

set_id = sys.argv[1]
max_features = 30000
learning_rate = 0.0005
maxlen = 25
batch_size = 32
folds = 4
seed = 1234
epochs = 2

# Preprocessing data sets
def get_padded_dataset(dataset, tokenizer=None):
    labels = [ x['label'] for x in dataset]
    data = [ x['text'] for x in dataset]

    # Preprocessing text
    if tokenizer is None:
        print('construct tokenizer')
        tokenizer = Tokenizer(num_words=max_features)
        tokenizer.fit_on_texts(data)
    
    data_seqs = tokenizer.texts_to_sequences(data)
    data_seqs_padded = pad_sequences(data_seqs, maxlen=maxlen)
    labels = np.array(labels)
    return data_seqs_padded, labels, tokenizer

print('Loading data...')
all_data = read_file(set_id)
all_pred, all_true = [], []

fold_no = 1
for train, dev, test in fold_iterator(all_data, K=folds, random_seed=seed):
    print("fold", fold_no)
    fold_no += 1

    x_train, y_train, tokenizer = get_padded_dataset(train)
    x_dev, y_dev, tokenizer = get_padded_dataset(dev, tokenizer=tokenizer)
    x_test, y_test, tokenizer = get_padded_dataset(test, tokenizer=tokenizer)
    print(len(tokenizer.word_docs), 'unique tokens')
    print(x_train.shape, 'train sequences')
    print(x_test.shape, 'test sequences')
    print(x_dev.shape, 'dev sequences')

    print('Build model...')

    sequence_input = Input(shape=(maxlen,), dtype='int32')
    embedding_layer = Embedding(max_features,
                                128,
                                input_length=maxlen)(sequence_input)

    x = Bidirectional(GRU(128, return_sequences=True, dropout=0.1, recurrent_dropout=0.1))(embedding_layer)
    x = Bidirectional(GRU(64, return_sequences=True, dropout=0.1, recurrent_dropout=0.1))(x)
    x = AttentionWithContext()(x)
    x = Dense(64, activation='relu')(x)
    x = Dense(1, activation='sigmoid')(x)
    model = Model(sequence_input, x)

    opt = TFOptimizer(tf.keras.optimizers.Adam())
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['binary_accuracy'])

    print('Train...')
    model.fit(x_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            shuffle=True,
            validation_data=(x_dev, y_dev))

    y_pred = model.predict(x_test)

    y_pred = [ 1 if s >= 0.5 else 0 for s in y_pred ]
    all_true += list(y_test)
    all_pred += y_pred

    print(classification_report(y_test, y_pred))

print("Total evaluation:\n", classification_report(all_true, all_pred))
print(confusion_matrix(all_true, all_pred))