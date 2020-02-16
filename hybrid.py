from keras.layers import Dense, Embedding, GlobalMaxPooling1D, CuDNNGRU, GRU, LSTM, Conv1D, Conv2D, MaxPooling1D, MaxPool2D, Flatten, Dropout, Concatenate, Reshape, GlobalAveragePooling1D
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.utils import to_categorical
from keras.optimizers import Nadam
from keras.engine.input_layer import Input
from keras.models import Model
from keras_attention import AttentionWithContext
from keras.layers.wrappers import Bidirectional
from keras.callbacks.callbacks import EarlyStopping
from sklearn.metrics import classification_report, confusion_matrix
from data import *
import tensorflow as tf
from keras.optimizers import TFOptimizer
import sys
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

set_id = sys.argv[1]
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

def model_hybrid():
    filter_sizes = [1,2,3,5]
    num_filters = 36

    inp_1 = Input(shape=(maxcharlen,))
    e_x = Embedding(max_char_features, char_embed_size)(inp_1)
    r_x = Reshape((maxcharlen, char_embed_size, 1))(e_x)

    c_maxpool_pool = []
    c_atts = []
    for i in range(len(filter_sizes)):
        conv = Conv2D(num_filters, kernel_size=(filter_sizes[i], char_embed_size),
                                     kernel_initializer='he_normal', activation='relu')(r_x)
        c_maxpool_pool.append(MaxPool2D(pool_size=(maxcharlen - filter_sizes[i] + 1, 1))(conv))
        conv = Conv1D(num_filters, kernel_size=(filter_sizes[i]),
                                     kernel_initializer='he_normal', activation='relu')(e_x)
        c_atts.append(AttentionWithContext()(conv))

    z = Concatenate(axis=1)(c_maxpool_pool)   
    z = Flatten()(z)
    c_z = Dropout(0.1)(z)

    inp_2 = Input(shape=(maxlen,))
    e_w = Embedding(max_features, embed_size)(inp_2)
    r_w = Reshape((maxlen, embed_size, 1))(e_w)

    maxpool_pool = []
    atts = []
    for i in range(len(filter_sizes)):
        conv = Conv2D(num_filters, kernel_size=(filter_sizes[i], embed_size),
                                     kernel_initializer='he_normal', activation='relu')(r_w)
        maxpool_pool.append(MaxPool2D(pool_size=(maxlen - filter_sizes[i] + 1, 1))(conv))
        conv = Conv1D(num_filters, kernel_size=(filter_sizes[i]),
                                     kernel_initializer='he_normal', activation='relu')(e_w)
        atts.append(AttentionWithContext()(conv))

    z = Concatenate(axis=1)(maxpool_pool)   
    z = Flatten()(z)
    z = Dropout(0.1)(z)

    x = Bidirectional(GRU(128, return_sequences=True, recurrent_dropout=0.1))(e_w)
    a_1 = AttentionWithContext()(x)
    x = Bidirectional(GRU(64, return_sequences=True, recurrent_dropout=0.1))(x)
    avg_pool = GlobalAveragePooling1D()(x)
    max_pool = GlobalMaxPooling1D()(x)
    a_2 = AttentionWithContext()(x)

    conc = Concatenate(axis=1)([avg_pool, max_pool, a_1, a_2, c_z, z] + atts + c_atts)
    outp = Dense(64, activation="relu")(conc)
    outp = Dropout(0.1)(outp)
    outp = Dense(1, activation="sigmoid")(outp)

    opt = TFOptimizer(tf.keras.optimizers.Adam(learning_rate=learning_rate))
    model = Model(inputs=[inp_1, inp_2], outputs=outp)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    
    return model


def model_cnn():
    filter_sizes = [1,2,3,5]
    num_filters = 36

    inp = Input(shape=(maxlen,))
    x = Embedding(max_features, embed_size)(inp)
    x = Reshape((maxlen, embed_size, 1))(x)

    maxpool_pool = []
    for i in range(len(filter_sizes)):
        conv = Conv2D(num_filters, kernel_size=(filter_sizes[i], embed_size),
                                     kernel_initializer='he_normal', activation='relu')(x)
        maxpool_pool.append(MaxPool2D(pool_size=(maxlen - filter_sizes[i] + 1, 1))(conv))

    z = Concatenate(axis=1)(maxpool_pool)   
    z = Flatten()(z)
    z = Dropout(0.1)(z)

    outp = Dense(1, activation="sigmoid")(z)

    model = Model(inputs=inp, outputs=outp)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    return model


# Preprocessing data sets
def get_padded_dataset(dataset, _tokenizer=None, char_level=False):
    labels = [ x['label'] for x in dataset]
    data = [ x['text'] for x in dataset]

    # Preprocessing text
    if _tokenizer is None:
        print('construct tokenizer')
        if char_level:
            _tokenizer = Tokenizer(num_words=max_char_features, char_level=char_level)
        else:
            _tokenizer = Tokenizer(num_words=max_features)
        _tokenizer.fit_on_texts(data)
    
    data_seqs = _tokenizer.texts_to_sequences(data)
    data_seqs_padded = pad_sequences(data_seqs, maxlen=( maxcharlen if char_level else maxlen))
    labels = np.array(labels)
    return data_seqs_padded, labels, _tokenizer


print('Loading data...')
all_data = read_file(set_id)
all_data = read_file(set_id)
all_pred, all_true = [], []


fold_no = 1
for train, dev, test in fold_iterator(all_data, K=folds, dev_ratio=0.1, random_seed=seed):
    print("fold", fold_no)
    fold_no += 1

    x_train, y_train, tokenizer = get_padded_dataset(train)
    x_c_train, _, ctokenizer = get_padded_dataset(train, char_level=True)
    x_dev, y_dev, tokenizer = get_padded_dataset(dev, _tokenizer=tokenizer)
    x_c_dev, _, ctokenizer = get_padded_dataset(dev, _tokenizer=ctokenizer, char_level=True)
    x_test, y_test, tokenizer = get_padded_dataset(test, _tokenizer=tokenizer)
    x_c_test, _, ctokenizer = get_padded_dataset(test, _tokenizer=ctokenizer, char_level=True)
    
    print(len(tokenizer.word_docs), 'unique tokens')
    print(len(ctokenizer.word_docs), 'unique char')
    print(x_train.shape[0], 'train sequences')
    print(x_dev.shape[0], 'dev sequences')
    print(x_test.shape[0], 'test sequences')

    print('Build model...')

    # model = model_hybrid()
    model = model_cnn()

    # print('Train...')
    # model.fit([x_c_train, x_train], y_train,
    #         batch_size=batch_size,
    #         epochs=epochs,
    #         validation_data=([x_c_dev, x_dev], y_dev))

    print('Train...')
    model.fit(x_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(x_dev, y_dev))

    # y_pred = model.predict([x_c_test, x_test])
    y_pred = model.predict(x_test)
    y_pred = [ 1 if s >= 0.5 else 0 for s in y_pred ]
    all_true += list(y_test)
    all_pred += y_pred

    print(classification_report(y_test, y_pred))

print("Total evaluation:\n", classification_report(all_true, all_pred))
print(confusion_matrix(all_true, all_pred))