from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import Dense, Embedding, CuDNNGRU, GRU, LSTM, Concatenate
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
import json

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = "1,7"

set_id = sys.argv[1]
max_features = 30000
learning_rate = 0.001
maxlen = 64
batch_size = 32
folds = 4
seed = 1234
epochs = 2


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
for train, dev, test in fold_iterator(all_data, K=folds, random_seed=seed):
    print("fold", fold_no)
    fold_no += 1

    x_train, y_train, tokenizer = get_padded_dataset(train)
    x_dev, y_dev, tokenizer = get_padded_dataset(dev, _tokenizer=tokenizer)
    x_test, y_test, tokenizer = get_padded_dataset(test, _tokenizer=tokenizer)
    print(len(tokenizer.word_docs), 'unique tokens')
    print(x_train.shape, 'train sequences')
    print(x_test.shape, 'test sequences')
    print(x_dev.shape, 'dev sequences')

    print('Build model...')

    emb_index = get_word_index(set_id)
    embedding_matrix = get_embeddings(set_id)[[0, ] + list(emb_index.get(k, 0) for k in tokenizer.word_index)]
    # concatenated_embed_size = embedding_matrix.shape[1] + embed_size
    sequence_input = Input(shape=(maxlen,), dtype='int32')
    em1 = Embedding(max_features,
                    256,
                    input_length=maxlen)(sequence_input)
    em2 = Embedding(embedding_matrix.shape[0], embedding_matrix.shape[1], weights=[embedding_matrix],
                    trainable=False)(sequence_input)

    x = Concatenate(axis=2)([em1, em2])
    x = Bidirectional(GRU(128, return_sequences=True, dropout=0.1, recurrent_dropout=0.1))(x)
    x = Bidirectional(GRU(64, return_sequences=True, dropout=0.1, recurrent_dropout=0.1))(x)
    x = AttentionWithContext()(x)
    x = Dense(64, activation='relu')(x)
    x = Dense(1, activation='sigmoid')(x)
    model = Model(sequence_input, x)

    opt = TFOptimizer(tf.keras.optimizers.Adam(lr=learning_rate))
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['binary_accuracy'])

    print('Train...')
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              shuffle=True,
              validation_data=(x_dev, y_dev))

    y_pred = model.predict(x_test)
    # save these probabilities into a list to save them in a file
    all_probs += y_pred.flatten().tolist()

    y_pred = [1 if s >= 0.5 else 0 for s in y_pred]
    all_true += list(y_test)
    all_pred += y_pred

    print(classification_report(y_test, y_pred))

# save the probs to a file
np.save("bi_gru_probs_emb", all_probs)

print("Total evaluation:\n", classification_report(all_true, all_pred))
print(confusion_matrix(all_true, all_pred))
