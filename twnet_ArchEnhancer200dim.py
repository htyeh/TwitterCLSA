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
import pickle, json
import utils
from keras import optimizers
import keras.backend as K

train_dir = './TWEETS/CLEAN/EN_CLARIN_full/train'
dev_dir = './TWEETS/CLEAN/EN_CLARIN_full/dev'
test_dir = './TWEETS/CLEAN/EN_CLARIN_full/test'
de_train_dir = './TWEETS/CLEAN/ES_CLARIN_300/train'
de_dev_dir = './TWEETS/CLEAN/ES_CLARIN_300/dev'
de_test_dir = './TWEETS/CLEAN/ES_CLARIN_300/test'
train_texts, train_labels = utils.load_data(train_dir)
dev_texts, dev_labels = utils.load_data(dev_dir)
test_texts, test_labels = utils.load_data(test_dir)
de_train_texts, de_train_labels = utils.load_data(de_train_dir)
de_dev_texts, de_dev_labels = utils.load_data(de_dev_dir)
de_test_texts, de_test_labels = utils.load_data(de_test_dir)

# MAX_WORDS = 30000
MAXLEN = 30    # max tweet word count

tokenizer = Tokenizer()
tokenizer.fit_on_texts(train_texts + dev_texts + test_texts + de_train_texts + de_dev_texts + de_test_texts)

vocab_size = len(tokenizer.word_index) + 1  # +UNK
print('unique tokens in tokenizer: ' + str(vocab_size - 1))

print('transforming into vectors...')
train_sequences = tokenizer.texts_to_sequences(train_texts)
dev_sequences = tokenizer.texts_to_sequences(dev_texts)
test_sequences = tokenizer.texts_to_sequences(test_texts)
de_train_sequences = tokenizer.texts_to_sequences(de_train_texts)
de_dev_sequences = tokenizer.texts_to_sequences(de_dev_texts)
de_test_sequences = tokenizer.texts_to_sequences(de_test_texts)
print('padding to ' + str(MAXLEN) + ' words each...')
train_data = pad_sequences(train_sequences, maxlen=MAXLEN)
dev_data = pad_sequences(dev_sequences, maxlen=MAXLEN)
test_data = pad_sequences(test_sequences, maxlen=MAXLEN)
de_train_data = pad_sequences(de_train_sequences, maxlen=MAXLEN)
de_dev_data = pad_sequences(de_dev_sequences, maxlen=MAXLEN)
de_test_data = pad_sequences(de_test_sequences, maxlen=MAXLEN)
train_labels = np.asarray(train_labels)
dev_labels = np.asarray(dev_labels)
test_labels = np.asarray(test_labels)
de_train_labels = np.asarray(de_train_labels)
de_dev_labels = np.asarray(de_dev_labels)
de_test_labels = np.asarray(de_test_labels)

print('en train data tensor shape = ', train_data.shape)
print('en train label tensor shape = ', train_labels.shape)
print('en dev data tensor shape = ', dev_data.shape)
print('en dev label tensor shape = ', dev_labels.shape)
print('en test data tensor shape = ', test_data.shape)
print('en test label tensor shape = ', test_labels.shape)
print('de train data tensor shape = ', de_train_data.shape)
print('de train label tensor shape = ', de_train_labels.shape)
print('de dev data tensor shape = ', de_dev_data.shape)
print('de dev label tensor shape = ', de_dev_labels.shape)
print('de test data tensor shape = ', de_test_data.shape)
print('de test label tensor shape = ', de_test_labels.shape)

train_data, train_labels = utils.shuffle(train_data, train_labels)
dev_data, dev_labels = utils.shuffle(dev_data, dev_labels)
test_data, test_labels = utils.shuffle(test_data, test_labels)
de_train_data, de_train_labels = utils.shuffle(de_train_data, de_train_labels)
de_dev_data, de_dev_labels = utils.shuffle(de_dev_data, de_dev_labels)
de_test_data, de_test_labels = utils.shuffle(de_test_data, de_test_labels)

x_train = train_data
y_train = train_labels
x_val = dev_data
y_val = dev_labels
x_test = test_data
y_test = test_labels
x_train_de = de_train_data
y_train_de = de_train_labels
x_val_de = de_dev_data
y_val_de = de_dev_labels
x_test_de = de_test_data
y_test_de = de_test_labels

# tests
print(x_train_de[:3])
print(x_test_de[:3])

EMBEDDING_DIM = 100

embeddings_index = utils.load_embs_2_dict('EMBEDDINGS/EN_ES.txt.w2v')

embedding_matrix = utils.build_emb_matrix(num_embedding_vocab=vocab_size, embedding_dim=EMBEDDING_DIM, word_index=tokenizer.word_index, embeddings_index=embeddings_index)

global_en_mic_train = 0
global_de_mic_train = 0
global_en_mac_train = 0
global_de_mac_train = 0
global_en_mic_tune = 0
global_de_mic_tune = 0
global_en_mac_tune = 0
global_de_mac_tune = 0
num_iterations = 8

for i in range(num_iterations):
    print('training iteration:', i + 1)

    # optimize architecture
    print('optimizing architecture...')
    input_layer = layers.Input(shape=(MAXLEN,))
    emb1out = layers.Embedding(vocab_size, EMBEDDING_DIM, weights=[embedding_matrix], trainable=False, input_length=MAXLEN)(input_layer)
    emb2out = layers.Embedding(vocab_size, EMBEDDING_DIM, weights=[embedding_matrix], trainable=False, input_length=MAXLEN)(input_layer)
    merged_embs = layers.concatenate([emb1out, emb2out])
    bilstm_out = layers.Bidirectional(layers.LSTM(128))(merged_embs)
    dropout = layers.Dropout(0.2)(bilstm_out)
    dense1 = layers.Dense(64, activation='relu')(dropout)
    dense2 = layers.Dense(64, activation='relu')(dense1)
    output_layer = layers.Dense(3, activation='softmax')(dense2)
    model = models.Model(inputs=input_layer, outputs=output_layer)
    Adam = optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['acc'])
    print(model.summary())
    print('LR:', K.eval(model.optimizer.lr))
    es = EarlyStopping(monitor='val_loss', mode='auto', min_delta=0, patience=5, restore_best_weights=True, verbose=1)
    mc = ModelCheckpoint('best_model.h5', monitor='val_loss', mode='auto', verbose=1, save_best_only=True)
    model.fit(x_train, y_train, validation_data=(x_val, y_val), batch_size=64, epochs=100, shuffle=True, callbacks=[es, mc])
    # print('trained embedding shape:', model.layers[0].get_weights()[0].shape)

    # tune EmbLayer
    print('tuning embs...')
    model2 = models.load_model('best_model.h5', compile=False)
    model2.layers[2].trainable = True
    for layer in model2.layers[4:]:
        layer.trainable = False
    Adam = optimizers.Adam(learning_rate=0.0001)
    model2.compile(optimizer=Adam, loss='sparse_categorical_crossentropy', metrics=['acc'])
    print(model2.summary())
    print('LR:', K.eval(model2.optimizer.lr))
    es = EarlyStopping(monitor='val_loss', mode='auto', min_delta=0, patience=3, restore_best_weights=True, verbose=1)
    mc = ModelCheckpoint('best_model.h5', monitor='val_loss', mode='auto', verbose=1, save_best_only=True)
    model2.fit(x_train, y_train, validation_data=(x_val, y_val), batch_size=64, epochs=100, shuffle=True, callbacks=[es, mc])
    # print('trained embedding shape:', model.layers[0].get_weights()[0].shape)

    gold_en = y_test
    predicted_en = model2.predict(x_test).argmax(axis=1)
    gold_de = y_test_de
    predicted_de = model2.predict(x_test_de).argmax(axis=1)

    en_mic, de_mic, en_mac, de_mac = utils.test_evaluation(gold_en, predicted_en, gold_de, predicted_de)
    global_en_mic_train += en_mic
    global_de_mic_train += de_mic
    global_en_mac_train += en_mac
    global_de_mac_train += de_mac

    # de fine-tuning
    FINETUNE = True
    if FINETUNE:
        print('performing classical fine-tuning...')
        print('train:', de_train_dir)
        print('dev:', de_dev_dir)
        print('fine-tuning architecture...')
        model3 = models.load_model('best_model.h5', compile=False)
        model3.layers[2].trainable = False  # freeze one EmbLayer
        for layer in model3.layers[4:]:
            layer.trainable = True  # unfreeze rest layers
        Adam = optimizers.Adam(learning_rate=0.0001)
        model3.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['acc'])
        print(model3.summary())
        print(K.eval(model3.optimizer.lr))
        es = EarlyStopping(monitor='val_loss', mode='auto', min_delta=0, patience=5, restore_best_weights=True, verbose=1)
        mc = ModelCheckpoint('best_model.h5', monitor='val_loss', mode='auto', verbose=1, save_best_only=True, save_weights_only=False)
        model3.fit(x_train_de, y_train_de, validation_data=(x_val_de, y_val_de), batch_size=64, epochs=100, shuffle=True, callbacks=[es, mc])

        print('fine-tuning embs...')
        model4 = models.load_model('best_model.h5', compile=False)
        model4.layers[2].trainable = True
        for layer in model4.layers[4:]:
            layer.trainable = False
        Adam = optimizers.Adam(learning_rate=0.0001)
        model4.compile(optimizer=Adam, loss='sparse_categorical_crossentropy', metrics=['acc'])
        print(model4.summary())
        print('LR:', K.eval(model4.optimizer.lr))
        es = EarlyStopping(monitor='val_loss', mode='auto', min_delta=0, patience=3, restore_best_weights=True, verbose=1)
        mc = ModelCheckpoint('best_model.h5', monitor='val_loss', mode='auto', verbose=1, save_best_only=True)
        model4.fit(x_train_de, y_train_de, validation_data=(x_val_de, y_val_de), batch_size=64, epochs=100, shuffle=True, callbacks=[es, mc])

        gold_en = y_test
        predicted_en = model4.predict(x_test).argmax(axis=1)
        gold_de = y_test_de
        predicted_de = model4.predict(x_test_de).argmax(axis=1)

        en_mic, de_mic, en_mac, de_mac = utils.test_evaluation(gold_en, predicted_en, gold_de, predicted_de)
        global_en_mic_tune += en_mic
        global_de_mic_tune += de_mic
        global_en_mac_tune += en_mac
        global_de_mac_tune += de_mac

print()
print('AVG OF', num_iterations, 'TRAIN-ITERATIONS')
en_micro_train = round( (global_en_mic_train/num_iterations), 4)
de_micro_train = round( (global_de_mic_train/num_iterations), 4)
en_macro_train = round( (global_en_mac_train/num_iterations), 4)
de_macro_train = round( (global_de_mac_train/num_iterations), 4)
print('{0: <10}'.format('En-micro') + '\t' + '{0: <10}'.format('De-micro') + '\t' + '{0: <10}'.format('En-macro') + '\t' + '{0: <10}'.format('De-macro'))
print('{0: <10}'.format(en_micro_train) + '\t' + '{0: <10}'.format(de_micro_train) + '\t' + '{0: <10}'.format(en_macro_train) + '\t' + '{0: <10}'.format(de_macro_train))

if FINETUNE:
    print('AVG OF', num_iterations, 'TUNE-ITERATIONS')
    en_micro_tune = round( (global_en_mic_tune/num_iterations), 4)
    de_micro_tune = round( (global_de_mic_tune/num_iterations), 4)
    en_macro_tune = round( (global_en_mac_tune/num_iterations), 4)
    de_macro_tune = round( (global_de_mac_tune/num_iterations), 4)
    print('{0: <10}'.format('En-micro') + '\t' + '{0: <10}'.format('De-micro') + '\t' + '{0: <10}'.format('En-macro') + '\t' + '{0: <10}'.format('De-macro'))
    print('{0: <10}'.format(en_micro_tune) + '\t' + '{0: <10}'.format(de_micro_tune) + '\t' + '{0: <10}'.format(en_macro_tune) + '\t' + '{0: <10}'.format(de_macro_tune))

# toy tests
# toy_sents = tokenizer.texts_to_sequences(['the cat sat on the mat', 'what a great movie', 'better not again', 'terrible, worst ever', 'best film ever', 'today is Tuesday'])
# toy_data = pad_sequences(toy_sents, maxlen=MAXLEN)
# toy_gold = [1, 2, 0, 0, 2, 1]
# prediction = model.predict(toy_data)
# print(toy_gold)
# print(prediction.argmax(axis=1))

# plot results
# utils.plot(history)