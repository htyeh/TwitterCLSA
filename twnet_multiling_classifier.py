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
import utils
from keras import optimizers
import keras.backend as K

en_train_dir = './TWEETS/CLEAN/MULTILING_CLASSIFIER/EN/train'
en_dev_dir = './TWEETS/CLEAN/MULTILING_CLASSIFIER/EN/dev'
en_test_dir = './TWEETS/CLEAN/MULTILING_CLASSIFIER/EN/test'
en_train_texts, en_train_labels = utils.load_data(en_train_dir)
en_dev_texts, en_dev_labels = utils.load_data(en_dev_dir)
en_test_texts, en_test_labels = utils.load_data(en_test_dir)

de_train_dir = './TWEETS/CLEAN/MULTILING_CLASSIFIER/DE/train'
de_test_dir = './TWEETS/CLEAN/MULTILING_CLASSIFIER/DE/test'
de_train_texts, de_train_labels = utils.load_data(de_train_dir)
de_test_texts, de_test_labels = utils.load_data(de_test_dir)

es_train_dir = './TWEETS/CLEAN/MULTILING_CLASSIFIER/ES/train'
es_test_dir = './TWEETS/CLEAN/MULTILING_CLASSIFIER/ES/test'
es_train_texts, es_train_labels = utils.load_data(es_train_dir)
es_test_texts, es_test_labels = utils.load_data(es_test_dir)

hu_train_dir = './TWEETS/CLEAN/MULTILING_CLASSIFIER/HU/train'
hu_test_dir = './TWEETS/CLEAN/MULTILING_CLASSIFIER/HU/test'
hu_train_texts, hu_train_labels = utils.load_data(hu_train_dir)
hu_test_texts, hu_test_labels = utils.load_data(hu_test_dir)

sk_train_dir = './TWEETS/CLEAN/MULTILING_CLASSIFIER/SK/train'
sk_test_dir = './TWEETS/CLEAN/MULTILING_CLASSIFIER/SK/test'
sk_train_texts, sk_train_labels = utils.load_data(sk_train_dir)
sk_test_texts, sk_test_labels = utils.load_data(sk_test_dir)

sv_train_dir = './TWEETS/CLEAN/MULTILING_CLASSIFIER/SV/train'
sv_test_dir = './TWEETS/CLEAN/MULTILING_CLASSIFIER/SV/test'
sv_train_texts, sv_train_labels = utils.load_data(sv_train_dir)
sv_test_texts, sv_test_labels = utils.load_data(sv_test_dir)

it_train_dir = './TWEETS/CLEAN/MULTILING_CLASSIFIER/IT/train'
it_test_dir = './TWEETS/CLEAN/MULTILING_CLASSIFIER/IT/test'
it_train_texts, it_train_labels = utils.load_data(it_train_dir)
it_test_texts, it_test_labels = utils.load_data(it_test_dir)

pt_train_dir = './TWEETS/CLEAN/MULTILING_CLASSIFIER/PT/train'
pt_test_dir = './TWEETS/CLEAN/MULTILING_CLASSIFIER/PT/test'
pt_train_texts, pt_train_labels = utils.load_data(pt_train_dir)
pt_test_texts, pt_test_labels = utils.load_data(pt_test_dir)

# MAX_WORDS = 30000
MAXLEN = 30    # max tweet word count

tokenizer = Tokenizer()
tokenizer.fit_on_texts(en_train_texts + en_dev_texts + en_test_texts + de_train_texts + de_test_texts + es_train_texts + es_test_texts + hu_train_texts + hu_test_texts + sk_train_texts + sk_test_texts + sv_train_texts + sv_test_texts + it_train_texts + it_test_texts + pt_train_texts + pt_test_texts)

vocab_size = len(tokenizer.word_index) + 1  # +UNK
print('unique tokens in tokenizer: ' + str(vocab_size - 1))

print('transforming into vectors...')
en_train_sequences = tokenizer.texts_to_sequences(en_train_texts)
en_dev_sequences = tokenizer.texts_to_sequences(en_dev_texts)
en_test_sequences = tokenizer.texts_to_sequences(en_test_texts)

de_train_sequences = tokenizer.texts_to_sequences(de_train_texts)
de_test_sequences = tokenizer.texts_to_sequences(de_test_texts)

es_train_sequences = tokenizer.texts_to_sequences(es_train_texts)
es_test_sequences = tokenizer.texts_to_sequences(es_test_texts)

hu_train_sequences = tokenizer.texts_to_sequences(hu_train_texts)
hu_test_sequences = tokenizer.texts_to_sequences(hu_test_texts)

sk_train_sequences = tokenizer.texts_to_sequences(sk_train_texts)
sk_test_sequences = tokenizer.texts_to_sequences(sk_test_texts)

sv_train_sequences = tokenizer.texts_to_sequences(sv_train_texts)
sv_test_sequences = tokenizer.texts_to_sequences(sv_test_texts)

it_train_sequences = tokenizer.texts_to_sequences(it_train_texts)
it_test_sequences = tokenizer.texts_to_sequences(it_test_texts)

pt_train_sequences = tokenizer.texts_to_sequences(pt_train_texts)
pt_test_sequences = tokenizer.texts_to_sequences(pt_test_texts)

print('padding to ' + str(MAXLEN) + ' words each...')
en_train_data = pad_sequences(en_train_sequences, maxlen=MAXLEN)
en_dev_data = pad_sequences(en_dev_sequences, maxlen=MAXLEN)
en_test_data = pad_sequences(en_test_sequences, maxlen=MAXLEN)

de_train_data = pad_sequences(de_train_sequences, maxlen=MAXLEN)
de_test_data = pad_sequences(de_test_sequences, maxlen=MAXLEN)

es_train_data = pad_sequences(es_train_sequences, maxlen=MAXLEN)
es_test_data = pad_sequences(es_test_sequences, maxlen=MAXLEN)

hu_train_data = pad_sequences(hu_train_sequences, maxlen=MAXLEN)
hu_test_data = pad_sequences(hu_test_sequences, maxlen=MAXLEN)

sk_train_data = pad_sequences(sk_train_sequences, maxlen=MAXLEN)
sk_test_data = pad_sequences(sk_test_sequences, maxlen=MAXLEN)

sv_train_data = pad_sequences(sv_train_sequences, maxlen=MAXLEN)
sv_test_data = pad_sequences(sv_test_sequences, maxlen=MAXLEN)

it_train_data = pad_sequences(it_train_sequences, maxlen=MAXLEN)
it_test_data = pad_sequences(it_test_sequences, maxlen=MAXLEN)

pt_train_data = pad_sequences(pt_train_sequences, maxlen=MAXLEN)
pt_test_data = pad_sequences(pt_test_sequences, maxlen=MAXLEN)

print('processing labels...')
en_train_labels = np.asarray(en_train_labels)
en_dev_labels = np.asarray(en_dev_labels)
en_test_labels = np.asarray(en_test_labels)

de_train_labels = np.asarray(de_train_labels)
de_test_labels = np.asarray(de_test_labels)

es_train_labels = np.asarray(es_train_labels)
es_test_labels = np.asarray(es_test_labels)

hu_train_labels = np.asarray(hu_train_labels)
hu_test_labels = np.asarray(hu_test_labels)

sk_train_labels = np.asarray(sk_train_labels)
sk_test_labels = np.asarray(sk_test_labels)

sv_train_labels = np.asarray(sv_train_labels)
sv_test_labels = np.asarray(sv_test_labels)

it_train_labels = np.asarray(it_train_labels)
it_test_labels = np.asarray(it_test_labels)

pt_train_labels = np.asarray(pt_train_labels)
pt_test_labels = np.asarray(pt_test_labels)

print('en train data tensor shape = ', en_train_data.shape)
print('en train label tensor shape = ', en_train_labels.shape)
print('en dev data tensor shape = ', en_dev_data.shape)
print('en dev label tensor shape = ', en_dev_labels.shape)
print('en test data tensor shape = ', en_test_data.shape)
print('en test label tensor shape = ', en_test_labels.shape)

print('de train data tensor shape = ', de_train_data.shape)
print('de train label tensor shape = ', de_train_labels.shape)
print('de test data tensor shape = ', de_test_data.shape)
print('de test label tensor shape = ', de_test_labels.shape)

print('es train data tensor shape = ', es_train_data.shape)
print('es train label tensor shape = ', es_train_labels.shape)
print('es test data tensor shape = ', es_test_data.shape)
print('es test label tensor shape = ', es_test_labels.shape)

print('hu train data tensor shape = ', hu_train_data.shape)
print('hu train label tensor shape = ', hu_train_labels.shape)
print('hu test data tensor shape = ', hu_test_data.shape)
print('hu test label tensor shape = ', hu_test_labels.shape)

print('sk train data tensor shape = ', sk_train_data.shape)
print('sk train label tensor shape = ', sk_train_labels.shape)
print('sk test data tensor shape = ', sk_test_data.shape)
print('sk test label tensor shape = ', sk_test_labels.shape)

print('sv train data tensor shape = ', sv_train_data.shape)
print('sv train label tensor shape = ', sv_train_labels.shape)
print('sv test data tensor shape = ', sv_test_data.shape)
print('sv test label tensor shape = ', sv_test_labels.shape)

print('it train data tensor shape = ', it_train_data.shape)
print('it train label tensor shape = ', it_train_labels.shape)
print('it test data tensor shape = ', it_test_data.shape)
print('it test label tensor shape = ', it_test_labels.shape)

print('pt train data tensor shape = ', pt_train_data.shape)
print('pt train label tensor shape = ', pt_train_labels.shape)
print('pt test data tensor shape = ', pt_test_data.shape)
print('pt test label tensor shape = ', pt_test_labels.shape)

en_train_data, en_train_labels = utils.shuffle(en_train_data, en_train_labels)
en_dev_data, en_dev_labels = utils.shuffle(en_dev_data, en_dev_labels)
en_test_data, en_test_labels = utils.shuffle(en_test_data, en_test_labels)

de_train_data, de_train_labels = utils.shuffle(de_train_data, de_train_labels)
de_test_data, de_test_labels = utils.shuffle(de_test_data, de_test_labels)

es_train_data, es_train_labels = utils.shuffle(es_train_data, es_train_labels)
es_test_data, es_test_labels = utils.shuffle(es_test_data, es_test_labels)

hu_train_data, hu_train_labels = utils.shuffle(hu_train_data, hu_train_labels)
hu_test_data, hu_test_labels = utils.shuffle(hu_test_data, hu_test_labels)

sk_train_data, sk_train_labels = utils.shuffle(sk_train_data, sk_train_labels)
sk_test_data, sk_test_labels = utils.shuffle(sk_test_data, sk_test_labels)

sv_train_data, sv_train_labels = utils.shuffle(sv_train_data, sv_train_labels)
sv_test_data, sv_test_labels = utils.shuffle(sv_test_data, sv_test_labels)

it_train_data, it_train_labels = utils.shuffle(it_train_data, it_train_labels)
it_test_data, it_test_labels = utils.shuffle(it_test_data, it_test_labels)

pt_train_data, pt_train_labels = utils.shuffle(pt_train_data, pt_train_labels)
pt_test_data, pt_test_labels = utils.shuffle(pt_test_data, pt_test_labels)

# x_train = en_train_data
x_train = np.concatenate((en_train_data, de_train_data, es_train_data, hu_train_data, sk_train_data, sv_train_data, it_train_data, pt_train_data))
# y_train = en_train_labels
y_train = np.concatenate((en_train_labels, de_train_labels, es_train_labels, hu_train_labels, sk_train_labels, sv_train_labels, it_train_labels, pt_train_labels))
x_val = en_dev_data
y_val = en_dev_labels

x_test_en = en_test_data
y_test_en = en_test_labels
x_test_de = de_test_data
y_test_de = de_test_labels
x_test_es = es_test_data
y_test_es = es_test_labels
x_test_hu = hu_test_data
y_test_hu = hu_test_labels
x_test_sk = sk_test_data
y_test_sk = sk_test_labels
x_test_sv = sv_test_data
y_test_sv = sv_test_labels
x_test_it = it_test_data
y_test_it = it_test_labels
x_test_pt = pt_test_data
y_test_pt = pt_test_labels


EMBEDDING_DIM = 300

embeddings_index = utils.load_embs_2_dict('EMBEDDINGS/EN_DE_ES_HU_SK_SV_IT_PT.txt', dim=EMBEDDING_DIM)

embedding_matrix = utils.build_emb_matrix(num_embedding_vocab=vocab_size, embedding_dim=EMBEDDING_DIM, word_index=tokenizer.word_index, embeddings_index=embeddings_index)

global_en_mic_train = 0
global_en_mac_train = 0

global_de_mic_train = 0
global_de_mac_train = 0

global_es_mic_train = 0
global_es_mac_train = 0

global_hu_mic_train = 0
global_hu_mac_train = 0

global_sk_mic_train = 0
global_sk_mac_train = 0

global_sv_mic_train = 0
global_sv_mac_train = 0

global_it_mic_train = 0
global_it_mac_train = 0

global_pt_mic_train = 0
global_pt_mac_train = 0

num_iterations = 10

for i in range(num_iterations):
    print('training iteration:', i + 1)

    # build model
    model = models.Sequential()
    model.add(layers.Embedding(vocab_size, EMBEDDING_DIM, weights=[embedding_matrix], trainable=False, input_length=MAXLEN))
    model.add(layers.Bidirectional(layers.LSTM(128)))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(3, activation='softmax'))
    # Adam = optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['acc'])
    print(model.summary())
    print(K.eval(model.optimizer.lr))
    es = EarlyStopping(monitor='val_loss', mode='auto', min_delta=0, patience=5, restore_best_weights=True, verbose=1)
    mc = ModelCheckpoint('best_model.h5', monitor='val_loss', mode='auto', verbose=1, save_best_only=True)
    history = model.fit(x_train, y_train, validation_data=(x_val, y_val), batch_size=64, epochs=1000, shuffle=True, callbacks=[es, mc])
    print('trained embedding shape:', model.layers[0].get_weights()[0].shape)
    # utils.save_embs_2_file(model, 0, tokenizer.word_index)

    # test_loss, test_acc = model.evaluate(x_test, y_test)
    # print('test loss:', test_loss, 'test acc:', test_acc)
    gold_list = []
    pred_list = []

    gold_en = y_test_en
    predicted_en = model.predict(x_test_en).argmax(axis=1)
    gold_list.append(gold_en)
    pred_list.append(predicted_en)

    gold_de = y_test_de
    predicted_de = model.predict(x_test_de).argmax(axis=1)
    gold_list.append(gold_de)
    pred_list.append(predicted_de)

    gold_es = y_test_es
    predicted_es = model.predict(x_test_es).argmax(axis=1)
    gold_list.append(gold_es)
    pred_list.append(predicted_es)

    gold_hu = y_test_hu
    predicted_hu = model.predict(x_test_hu).argmax(axis=1)
    gold_list.append(gold_hu)
    pred_list.append(predicted_hu)

    gold_sk = y_test_sk
    predicted_sk = model.predict(x_test_sk).argmax(axis=1)
    gold_list.append(gold_sk)
    pred_list.append(predicted_sk)

    gold_sv = y_test_sv
    predicted_sv = model.predict(x_test_sv).argmax(axis=1)
    gold_list.append(gold_sv)
    pred_list.append(predicted_sv)

    gold_it = y_test_it
    predicted_it = model.predict(x_test_it).argmax(axis=1)
    gold_list.append(gold_it)
    pred_list.append(predicted_it)

    gold_pt = y_test_pt
    predicted_pt = model.predict(x_test_pt).argmax(axis=1)
    gold_list.append(gold_pt)
    pred_list.append(predicted_pt)

    micro_list, macro_list = utils.multiling_evaluation(gold_list, pred_list)

    global_en_mic_train += micro_list[0]
    global_en_mac_train += macro_list[0]

    global_de_mic_train += micro_list[1]
    global_de_mac_train += macro_list[1]

    global_es_mic_train += micro_list[2]
    global_es_mac_train += macro_list[2]

    global_hu_mic_train += micro_list[3]
    global_hu_mac_train += macro_list[3]

    global_sk_mic_train += micro_list[4]
    global_sk_mac_train += macro_list[4]

    global_sv_mic_train += micro_list[5]
    global_sv_mac_train += macro_list[5]

    global_it_mic_train += micro_list[6]
    global_it_mac_train += macro_list[6]

    global_pt_mic_train += micro_list[7]
    global_pt_mac_train += macro_list[7]

print()
print('AVG OF', num_iterations, 'TRAIN-ITERATIONS')
en_micro_train = round( (global_en_mic_train/num_iterations), 4)
en_macro_train = round( (global_en_mac_train/num_iterations), 4)
de_micro_train = round( (global_de_mic_train/num_iterations), 4)
de_macro_train = round( (global_de_mac_train/num_iterations), 4)
es_micro_train = round( (global_es_mic_train/num_iterations), 4)
es_macro_train = round( (global_es_mac_train/num_iterations), 4)
hu_micro_train = round( (global_hu_mic_train/num_iterations), 4)
hu_macro_train = round( (global_hu_mac_train/num_iterations), 4)
sk_micro_train = round( (global_sk_mic_train/num_iterations), 4)
sk_macro_train = round( (global_sk_mac_train/num_iterations), 4)
sv_micro_train = round( (global_sv_mic_train/num_iterations), 4)
sv_macro_train = round( (global_sv_mac_train/num_iterations), 4)
it_micro_train = round( (global_it_mic_train/num_iterations), 4)
it_macro_train = round( (global_it_mac_train/num_iterations), 4)
pt_micro_train = round( (global_pt_mic_train/num_iterations), 4)
pt_macro_train = round( (global_pt_mac_train/num_iterations), 4)

print('{0: <20}'.format('En (micro/macro)') + '\t' + '{0: <10}'.format(en_micro_train) + '\t' + '{0: <10}'.format(en_macro_train))
print('{0: <20}'.format('De (micro/macro)') + '\t' + '{0: <10}'.format(de_micro_train) + '\t' + '{0: <10}'.format(de_macro_train))
print('{0: <20}'.format('Es (micro/macro)') + '\t' + '{0: <10}'.format(es_micro_train) + '\t' + '{0: <10}'.format(es_macro_train))
print('{0: <20}'.format('Hu (micro/macro)') + '\t' + '{0: <10}'.format(hu_micro_train) + '\t' + '{0: <10}'.format(hu_macro_train))
print('{0: <20}'.format('Sk (micro/macro)') + '\t' + '{0: <10}'.format(sk_micro_train) + '\t' + '{0: <10}'.format(sk_macro_train))
print('{0: <20}'.format('Sv (micro/macro)') + '\t' + '{0: <10}'.format(sv_micro_train) + '\t' + '{0: <10}'.format(sv_macro_train))
print('{0: <20}'.format('It (micro/macro)') + '\t' + '{0: <10}'.format(it_micro_train) + '\t' + '{0: <10}'.format(it_macro_train))
print('{0: <20}'.format('Pt (micro/macro)') + '\t' + '{0: <10}'.format(pt_micro_train) + '\t' + '{0: <10}'.format(pt_macro_train))