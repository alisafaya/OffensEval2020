from keras.layers import Dense, Embedding, GlobalMaxPooling1D, CuDNNGRU, GRU, LSTM, Conv1D, Conv2D, MaxPooling1D, \
    MaxPool2D, Flatten, Dropout, Concatenate, Reshape, GlobalAveragePooling1D
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.utils import to_categorical
from keras.optimizers import Nadam
from keras.engine.input_layer import Input
from keras.models import Model
from sklearn.metrics import classification_report, confusion_matrix
from data import *
import tensorflow as tf
import numpy as np
from keras.optimizers import TFOptimizer
import sys
import os
import json

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = "7"

output_id = sys.argv[0].split('.')[0]
set_id = sys.argv[1]
# set_id = "da"
max_features = 30000
max_char_features = 512
learning_rate = 1e-3
embed_size = 256
char_embed_size = 512
batch_size = 32
maxlen = 32
maxcharlen = 128
epochs = 3
folds = 4
seed = 1234
output_path = './output/' + set_id + '/' + output_id


def model_cnn(embedding_matrix):
    filter_sizes = [1, 2, 3, 5]
    num_filters = 24

    inp = Input(shape=(maxlen,))
    em1 = Embedding(max_features, embed_size)(inp)
    em1 = Reshape((maxlen, embed_size, 1))(em1)

    if embedding_matrix is not None:
        concatenated_embed_size = embedding_matrix.shape[1] + embed_size
        em2 = Embedding(embedding_matrix.shape[0], embedding_matrix.shape[1], weights=[embedding_matrix],
                        trainable=False)(inp)
        x = Reshape((maxlen, embedding_matrix.shape[1], 1))(em2)
        x = Concatenate(axis=2)([em1, x])
    else:
        concatenated_embed_size = embed_size
        x = em1

    maxpool_pool = []

    # for i in range(len(filter_sizes)):
    #     conv = Conv2D(num_filters, kernel_size=(filter_sizes[i], embed_size),
    #                                  kernel_initializer='he_normal', activation='relu')(em1)
    #     maxpool_pool.append(MaxPool2D(pool_size=(maxlen - filter_sizes[i] + 1, 1))(conv))

    for i in range(len(filter_sizes)):
        conv = Conv2D(num_filters, kernel_size=(filter_sizes[i], concatenated_embed_size),
                      kernel_initializer='he_normal', activation='relu')(x)
        maxpool_pool.append(MaxPool2D(pool_size=(maxlen - filter_sizes[i] + 1, 1))(conv))

    z = Concatenate(axis=1)(maxpool_pool)
    z = Flatten()(z)
    z = Dropout(0.1)(z)

    outp = Dense(1, activation="sigmoid")(z)

    model = Model(inputs=inp, outputs=outp)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model


def get_embeddings(set_id):
    return np.load("vec/%s/%s.npy" % (set_id, set_id))


def get_word_index(set_id):
    return json.loads(open("vec/%s/%s_index.json" % (set_id, set_id), "r").read())


def get_padded_dataset(dataset, _tokenizer=None, char_level=False, use_index=False):
    labels = [x['label'] for x in dataset]
    data = [x['text'] for x in dataset]

    # Preprocessing text
    if _tokenizer is None:
        if char_level:
            _tokenizer = Tokenizer(num_words=max_char_features, char_level=char_level)
        else:
            _tokenizer = Tokenizer(num_words=max_features)

        if use_index:
            _tokenizer.word_index = get_word_index(set_id)
        else:
            _tokenizer.fit_on_texts(data)

    data_seqs = _tokenizer.texts_to_sequences(data)
    data_seqs_padded = pad_sequences(data_seqs, maxlen=(maxcharlen if char_level else maxlen))
    labels = np.array(labels)
    return data_seqs_padded, labels, _tokenizer


print('Loading data...')
all_data = read_file(set_id)
all_pred, all_true = [], []

# list to save probs for ensembling
all_probs = []

fold_no = 1
print("all Data size:" + str(len(all_data)))
for train, dev, test in fold_iterator_sklearn(all_data, K=folds, random_seed=seed):
    print("fold", fold_no)
    fold_no += 1

    x_train, y_train, tokenizer = get_padded_dataset(train)
    x_dev, y_dev, tokenizer = get_padded_dataset(dev, _tokenizer=tokenizer)
    x_test, y_test, tokenizer = get_padded_dataset(test, _tokenizer=tokenizer)

    print(x_train.shape[0], 'train sequences')
    print(x_dev.shape[0], 'dev sequences')
    print(x_test.shape[0], 'test sequences')

    print('Build model...')

    # emb_index = get_word_index(set_id)
    # emb_mat = get_embeddings(set_id)[[0, ] + list(emb_index.get(k, 0) for k in tokenizer.word_index)]
    # model = model_cnn(emb_mat)
    model = model_cnn(None)

    print('Train...')
    model.fit(x_train, y_train,
              epochs=epochs,
              batch_size=batch_size,
              validation_data=(x_dev, y_dev))

    y_pred = model.predict(x_test)

    # save these probabilities into a list to save them in a file
    all_probs += y_pred.flatten().tolist()

    y_pred = [1 if s >= 0.5 else 0 for s in y_pred]

    all_true += list(y_test)
    all_pred += y_pred

    print(classification_report(y_test, y_pred))

np.save(output_path + ".probs", np.array(all_probs))
np.save(output_path + ".gold", np.array(all_true))

print("Total evaluation:\n", classification_report(all_true, all_pred))
print(confusion_matrix(all_true, all_pred))
