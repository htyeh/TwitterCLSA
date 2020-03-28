#!/usr/local/bin/python3
import os
import sys
from sklearn.feature_extraction.text import TfidfVectorizer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation, Conv1D, GRU, CuDNNGRU, CuDNNLSTM, BatchNormalization
from keras.layers import Bidirectional, GlobalMaxPool1D, MaxPooling1D, Add, Flatten
from keras.layers import GlobalAveragePooling1D, GlobalMaxPooling1D, concatenate, SpatialDropout1D
from keras.models import Model, load_model
from keras import initializers, regularizers, constraints, optimizers, layers, callbacks
from keras import backend as K
from keras.engine import InputSpec, Layer
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, TensorBoard, Callback, EarlyStopping
import tensorflow as tf
import numpy as np
import utils

train_dir = './TWEETS/CLEAN/EN_CLARIN_downsampled/train'
dev_dir = './TWEETS/CLEAN/EN_CLARIN_full/dev'
test_dir = './TWEETS/CLEAN/EN_CLARIN_full/test'
de_test_dir = './TWEETS/CLEAN/DE_CLARIN_full/test'
train_texts, train_labels = utils.load_data(train_dir)
dev_texts, dev_labels = utils.load_data(dev_dir)
test_texts, test_labels = utils.load_data(test_dir)
de_test_texts, de_test_labels = utils.load_data(de_test_dir)

# MAX_WORDS = 30000
MAXLEN = 30    # max tweet word count
EMBEDDING_DIM = 100

# vectorize texts
print('transforming into vectors...')
tokenizer = Tokenizer()
tokenizer.fit_on_texts(train_texts + dev_texts + test_texts + de_test_texts)

vocab_size = len(tokenizer.word_index) + 1      # +UNK
train_sequences = tokenizer.texts_to_sequences(train_texts)
dev_sequences = tokenizer.texts_to_sequences(dev_texts)
test_sequences = tokenizer.texts_to_sequences(test_texts)
de_test_sequences = tokenizer.texts_to_sequences(de_test_texts)
print('unique tokens in tokenizer: ' + str(vocab_size - 1))
print('padding to ' + str(MAXLEN) + ' words each...')
train_data = pad_sequences(train_sequences, maxlen=MAXLEN)
dev_data = pad_sequences(dev_sequences, maxlen=MAXLEN)
test_data = pad_sequences(test_sequences, maxlen=MAXLEN)
de_test_data = pad_sequences(de_test_sequences, maxlen=MAXLEN)
train_labels = np.asarray(train_labels)
dev_labels = np.asarray(dev_labels)
test_labels = np.asarray(test_labels)
de_test_labels = np.asarray(de_test_labels)

print('train data tensor shape = ', train_data.shape)
print('train label tensor shape = ', train_labels.shape)
print('dev data tensor shape = ', dev_data.shape)
print('dev label tensor shape = ', dev_labels.shape)
print('test data tensor shape = ', test_data.shape)
print('test label tensor shape = ', test_labels.shape)
print('cross-lingual test data tensor shape = ', de_test_data.shape)
print('cross-lingual test label tensor shape = ', de_test_labels.shape)

train_data, train_labels = utils.shuffle(train_data, train_labels)
dev_data, dev_labels = utils.shuffle(dev_data, dev_labels)
test_data, test_labels = utils.shuffle(test_data, test_labels)
de_test_data, de_test_labels = utils.shuffle(de_test_data, de_test_labels)

x_train = train_data
y_train = train_labels
x_val = dev_data
y_val = dev_labels
x_test = test_data
y_test = test_labels
x_test2 = de_test_data
y_test2 = de_test_labels

# tests
print(x_train[:3])
print(x_test2[:3])

# load pre-trained embeddings (specify the embedding dimension)
embeddings_index = utils.load_embs_2_dict('EMBEDDINGS/EN_DE.txt.w2v')
# embeddings_index = utils.load_embs_2_dict('EMBEDDINGS/crosslingual_EN-DE_english_twitter_100d_weighted.txt.w2v')
# embeddings_index = utils.load_embs_2_dict('EMBEDDINGS/crosslingual_EN-DE_german_twitter_100d_weighted.txt.w2v')

embedding_matrix = utils.build_emb_matrix(num_embedding_vocab=vocab_size, embedding_dim=EMBEDDING_DIM, word_index=tokenizer.word_index, embeddings_index=embeddings_index)

# build model
def build_model1(lr=0.0, lr_d=0.0, units=0, spatial_dr=0.0, kernel_size1=3, kernel_size2=2, dense_units=128, dr=0.1, conv_size=32):
    file_path = "best_model.hdf5"
    check_point = ModelCheckpoint(file_path, monitor = "val_loss", verbose = 1,
                                  save_best_only = True, mode = "min")
    early_stop = EarlyStopping(monitor = "val_loss", mode = "min", patience = 3)
    
    inp = Input(shape = (MAXLEN,))
    x = Embedding(vocab_size, EMBEDDING_DIM, weights = [embedding_matrix], trainable = False)(inp)
    x1 = SpatialDropout1D(spatial_dr)(x)

    x_gru = Bidirectional(GRU(units, return_sequences = True))(x1)
    x1 = Conv1D(conv_size, kernel_size=kernel_size1, padding='valid', kernel_initializer='he_uniform')(x_gru)
    avg_pool1_gru = GlobalAveragePooling1D()(x1)
    max_pool1_gru = GlobalMaxPooling1D()(x1)
    
    x3 = Conv1D(conv_size, kernel_size=kernel_size2, padding='valid', kernel_initializer='he_uniform')(x_gru)
    avg_pool3_gru = GlobalAveragePooling1D()(x3)
    max_pool3_gru = GlobalMaxPooling1D()(x3)
    
    x_lstm = Bidirectional(LSTM(units, return_sequences = True))(x1)
    x1 = Conv1D(conv_size, kernel_size=kernel_size1, padding='valid', kernel_initializer='he_uniform')(x_lstm)
    avg_pool1_lstm = GlobalAveragePooling1D()(x1)
    max_pool1_lstm = GlobalMaxPooling1D()(x1)
    
    x3 = Conv1D(conv_size, kernel_size=kernel_size2, padding='valid', kernel_initializer='he_uniform')(x_lstm)
    avg_pool3_lstm = GlobalAveragePooling1D()(x3)
    max_pool3_lstm = GlobalMaxPooling1D()(x3)
    
    
    x = concatenate([avg_pool1_gru, max_pool1_gru, avg_pool3_gru, max_pool3_gru,
                    avg_pool1_lstm, max_pool1_lstm, avg_pool3_lstm, max_pool3_lstm])
    x = BatchNormalization()(x)
    x = Dropout(dr)(Dense(dense_units, activation='relu') (x))
    x = BatchNormalization()(x)
    x = Dropout(dr)(Dense(int(dense_units / 2), activation='relu') (x))
    x = Dense(3, activation = "softmax")(x)
    model = Model(inputs = inp, outputs = x)
    model.compile(loss="sparse_categorical_crossentropy", optimizer=Adam(lr=lr, decay=lr_d), metrics=["accuracy"])
    # model.compile(loss="sparse_categorical_crossentropy", optimizer='adam', metrics=["accuracy"])

    history = model.fit(x_train, y_train, batch_size = 64, epochs = 100, validation_data=(x_val, y_val), 
                        verbose = 1, shuffle=True, callbacks = [check_point, early_stop])
    model = load_model(file_path)
    return model

model = build_model1(lr = 1e-3, lr_d = 1e-10, units = 128, spatial_dr = 0.5, kernel_size1=4, kernel_size2=4, dense_units=64, dr=0.2, conv_size=32)



# test_loss, test_acc = model.evaluate(x_test, y_test)
# print('test loss:', test_loss, 'test acc:', test_acc)
gold = y_test
predicted = model.predict(x_test).argmax(axis=1)
gold2 = y_test2
predicted2 = model.predict(x_test2).argmax(axis=1)
utils.test_evaluation(gold, predicted)
utils.test_evaluation(gold2, predicted2)

# toy tests
toy_sents = tokenizer.texts_to_sequences(['the cat sat on the mat', 'what a great movie', 'better not again', 'terrible, worst ever', 'best film ever', 'today is Tuesday'])
toy_data = pad_sequences(toy_sents, maxlen=MAXLEN)
toy_gold = [1, 2, 0, 0, 2, 1]
prediction = model.predict(toy_data)
print(toy_gold)
print(prediction.argmax(axis=1))

# plot results
# utils.plot(history)