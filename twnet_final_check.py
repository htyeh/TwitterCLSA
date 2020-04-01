#!/usr/local/bin/python3
import os
import sys
from keras import models
from keras import layers
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np
from sklearn.metrics import f1_score
import pickle
import utils

# train_dir = './TWEETS/CLEAN/EN_CLARIN_full/train'
# dev_dir = './TWEETS/CLEAN/EN_CLARIN_full/dev'
test_dir = './TWEETS/CLEAN/EN_CLARIN_full/test'
# de_train_dir = './TWEETS/CLEAN/DE_CLARIN_full/train'
# de_dev_dir = './TWEETS/CLEAN/DE_CLARIN_full/dev'
de_test_dir = './TWEETS/CLEAN/DE_CLARIN_full/test'
# train_texts, train_labels = utils.load_data(train_dir)
# dev_texts, dev_labels = utils.load_data(dev_dir)
test_texts, test_labels = utils.load_data(test_dir)
# de_train_texts, de_train_labels = utils.load_data(de_train_dir)
# de_dev_texts, de_dev_labels = utils.load_data(de_dev_dir)
de_test_texts, de_test_labels = utils.load_data(de_test_dir)

# MAX_WORDS = 30000
MAXLEN = 30    # max tweet word count

with open('tokenizer.pickle', 'rb') as tokenizer_input:
    tokenizer = pickle.load(tokenizer_input)
print('restored Tokenizer object from twnet_en')
print('transforming into vectors...')

vocab_size = len(tokenizer.word_index) + 1  # +UNK
print('unique tokens in tokenizer: ' + str(vocab_size - 1))
# train_sequences = tokenizer.texts_to_sequences(train_texts)
# dev_sequences = tokenizer.texts_to_sequences(dev_texts)
test_sequences = tokenizer.texts_to_sequences(test_texts)
# de_train_sequences = tokenizer.texts_to_sequences(de_train_texts)
# de_dev_sequences = tokenizer.texts_to_sequences(de_dev_texts)
de_test_sequences = tokenizer.texts_to_sequences(de_test_texts)
print('padding to ' + str(MAXLEN) + ' words each...')
# train_data = pad_sequences(train_sequences, maxlen=MAXLEN)
# dev_data = pad_sequences(dev_sequences, maxlen=MAXLEN)
test_data = pad_sequences(test_sequences, maxlen=MAXLEN)
# de_train_data = pad_sequences(de_train_sequences, maxlen=MAXLEN)
# de_dev_data = pad_sequences(de_dev_sequences, maxlen=MAXLEN)
de_test_data = pad_sequences(de_test_sequences, maxlen=MAXLEN)
# train_labels = np.asarray(train_labels)
# dev_labels = np.asarray(dev_labels)
test_labels = np.asarray(test_labels)
# de_train_labels = np.asarray(de_train_labels)
# de_dev_labels = np.asarray(de_dev_labels)
de_test_labels = np.asarray(de_test_labels)

# print('en train data tensor shape = ', train_data.shape)
# print('en train label tensor shape = ', train_labels.shape)
# print('en dev data tensor shape = ', dev_data.shape)
# print('en dev label tensor shape = ', dev_labels.shape)
print('en test data tensor shape = ', test_data.shape)
print('en test label tensor shape = ', test_labels.shape)
# print('de train data tensor shape = ', de_train_data.shape)
# print('de train label tensor shape = ', de_train_labels.shape)
# print('de dev data tensor shape = ', de_dev_data.shape)
# print('de dev label tensor shape = ', de_dev_labels.shape)
print('de test data tensor shape = ', de_test_data.shape)
print('de test label tensor shape = ', de_test_labels.shape)

# train_data, train_labels = utils.shuffle(train_data, train_labels)
# dev_data, dev_labels = utils.shuffle(dev_data, dev_labels)
test_data, test_labels = utils.shuffle(test_data, test_labels)
# de_train_data, de_train_labels = utils.shuffle(de_train_data, de_train_labels)
# de_dev_data, de_dev_labels = utils.shuffle(de_dev_data, de_dev_labels)
de_test_data, de_test_labels = utils.shuffle(de_test_data, de_test_labels)

# x_train_de = de_train_data
# y_train_de = de_train_labels
# x_val_de = de_dev_data
# y_val_de = de_dev_labels
x_test_en = test_data
y_test_en = test_labels
x_test_de = de_test_data
y_test_de = de_test_labels

# tests
print(x_test_en[:3])
print(x_test_de[:3])

EMBEDDING_DIM = 100

# embeddings_index = utils.load_embs_2_dict('EMBEDDINGS/EN_DE.txt.w2v')
# embeddings_index = utils.load_embs_2_dict('EMBEDDINGS/crosslingual_EN-DE_english_twitter_100d_weighted.txt.w2v')
# embeddings_index = utils.load_embs_2_dict('EMBEDDINGS/glove.twitter.27B.200d.txt', dim=EMBEDDING_DIM)

# embedding_matrix = utils.build_emb_matrix(num_embedding_vocab=vocab_size, embedding_dim=EMBEDDING_DIM, word_index=tokenizer.word_index, embeddings_index=embeddings_index)

# build model
model = models.load_model('best_model.h5')
print(model.summary())
# es = EarlyStopping(monitor='val_loss', mode='auto', min_delta=0, patience=5, restore_best_weights=True, verbose=1)
# mc = ModelCheckpoint('best_model.h5', monitor='val_loss', mode='auto', verbose=1, save_best_only=True)
# history = model.fit(x_train_de, y_train_de, validation_data=(x_val_de, y_val_de), batch_size=64, epochs=100, shuffle=True, callbacks=[es, mc])
print('trained embedding shape:', model.layers[0].get_weights()[0].shape)

# test_loss, test_acc = model.evaluate(x_test, y_test)
# print('test loss:', test_loss, 'test acc:', test_acc)
gold_en = y_test_en
predicted_en = model.predict(x_test_en).argmax(axis=1)
gold_de = y_test_de
predicted_de = model.predict(x_test_de).argmax(axis=1)

print('sample en gold:', gold_en[:30])
print('sample en pred:', predicted_en[:30])
print('micro en:', f1_score(gold_en, predicted_en, average='micro'))
print('macro en:', f1_score(gold_en, predicted_en, average='macro'))

print('sample de gold:', gold_de[:30])
print('sample de pred:', predicted_de[:30])
print('micro de:', f1_score(gold_de, predicted_de, average='micro'))
print('macro de:', f1_score(gold_de, predicted_de, average='macro'))

# utils.test_evaluation(gold, predicted)
# utils.test_evaluation(gold2, predicted2)

# toy tests
# toy_sents = tokenizer.texts_to_sequences(['the cat sat on the mat', 'what a great movie', 'better not again', 'terrible, worst ever', 'best film ever', 'today is Tuesday'])
# toy_data = pad_sequences(toy_sents, maxlen=MAXLEN)
# toy_gold = [1, 2, 0, 0, 2, 1]
# prediction = model.predict(toy_data)
# print(toy_gold)
# print(prediction.argmax(axis=1))

# plot results
# utils.plot(history)