#!/usr/local/bin/python3
import os
import sys
from keras import models
from keras import layers
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
import numpy as np
import utils

# load texts and labels
train_dir = './TWEETS/CLEAN/EN_balanced_split/train'
test_dir = './TWEETS/CLEAN/EN_balanced_split/test'
train_texts = []
train_labels = []
test_texts = []
test_labels = []
for label_file in ['neg.tsv', 'neu.tsv', 'pos.tsv']:
    print('loading train/' + label_file + '...')
    with open(os.path.join(train_dir, label_file)) as f:
        for line in f:
            id, polarity, text = [ele.strip() for ele in line.split('\t')]
            train_texts.append(text)
            if polarity == 'negative':
                train_labels.append(0)
            elif polarity == 'neutral':
                train_labels.append(1)
            elif polarity == 'positive':
                train_labels.append(2)
for label_file in ['neg.tsv', 'neu.tsv', 'pos.tsv']:
    print('loading test/' + label_file + '...')
    with open(os.path.join(test_dir, label_file)) as f:
        for line in f:
            id, polarity, text = [ele.strip() for ele in line.split('\t')]
            test_texts.append(text)
            if polarity == 'negative':
                test_labels.append(0)
            elif polarity == 'neutral':
                test_labels.append(1)
            elif polarity == 'positive':
                test_labels.append(2)
train_test_texts = train_texts + test_texts

# define hyperparameters
# MAX_WORDS = 30000
MAXLEN = 30    # max tweet word count
EMBEDDING_DIM = 100
# total (en_imbalanced): 7689 neg/ 22189 neu/ 19606 pos
# total (en_balanced): 10000 neg/ 10000 neu/ 10000 pos
# total (de): 989 neg/ 4131 neu/ 1509 pos

# vectorize texts
print('transforming into vectors...')
tokenizer = Tokenizer()
tokenizer.fit_on_texts(train_test_texts)
vocab_size = len(tokenizer.word_index) + 1      # +UNK
train_sequences = tokenizer.texts_to_sequences(train_texts)
test_sequences = tokenizer.texts_to_sequences(test_texts)
# print('unique tokens found: ' + str(vocab_size - 1) + ', using most frequent ' + str(MAX_WORDS))
print('unique tokens found in train: ' + str(vocab_size - 1))
print('padding to ' + str(MAXLEN) + ' words each...')
train_data = pad_sequences(train_sequences, maxlen=MAXLEN)
test_data = pad_sequences(test_sequences, maxlen=MAXLEN)
train_labels = np.asarray(train_labels)
test_labels = np.asarray(test_labels)
# labels = to_categorical(labels)
print('train data tensor shape = ', train_data.shape)
print('test data tensor shape = ', test_data.shape)
print('train label tensor shape = ', train_labels.shape)
print('test label tensor shape = ', test_labels.shape)

# shuffle
train_data, train_labels = utils.shuffle(train_data, train_labels)
test_data, test_labels = utils.shuffle(test_data, test_labels)

# define training & eval sizes
x_train = train_data
y_train = train_labels
# x_val = test_data
# y_val = test_labels
x_test = test_data
y_test = test_labels

# load pre-trained embeddings (specify the embedding dimension)
# embeddings_index = utils.load_embs_2_dict('EMBEDDINGS/EN_DE.txt.w2v')
embeddings_index = utils.load_embs_2_dict('EMBEDDINGS/crosslingual_EN-DE_english_twitter_100d_weighted.txt.w2v')

num_embedding_vocab = vocab_size
embedding_matrix = utils.build_emb_matrix(num_embedding_vocab=num_embedding_vocab, embedding_dim=EMBEDDING_DIM, word_index=tokenizer.word_index, embeddings_index=embeddings_index)

# build model
model = models.Sequential()
# model.add(layers.Embedding(vocab_size, EMBEDDING_DIM, input_length=MAXLEN))
model.add(layers.Embedding(num_embedding_vocab, EMBEDDING_DIM, weights=[embedding_matrix], trainable=False, input_length=MAXLEN))
model.add(layers.Bidirectional(layers.LSTM(EMBEDDING_DIM)))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(3, activation='softmax'))
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['acc'])
history = model.fit(x_train, y_train, validation_split=0.1, batch_size=32, epochs=5, shuffle=True)
print('trained embedding shape:', model.layers[0].get_weights()[0].shape)

# substitude following 2 lines with evaluation function
test_loss, test_acc = model.evaluate(x_test, y_test)
print('test loss:', test_loss, 'test acc:', test_acc)
gold = y_test
predicted = model.predict(x_test).argmax(axis=1)
utils.test_evaluation(gold, predicted)

toy_sents = tokenizer.texts_to_sequences(['this morning I had a terrible pancake that I hated', 'wow what a great movie', 'you better not come again', 'terrible, worst ever', 'best film ever', 'today is Tuesday'])
toy_data = pad_sequences(toy_sents, maxlen=MAXLEN)
prediction = model.predict(toy_data)
print(prediction.argmax(axis=1))

# plot results
# utils.plot(history)

# model.add(layers.Conv1D(32, 7, activation='relu'))
# --Conv1D(filter_size(features), filter_height(num words each time), activation)
# --Conv1D output (batch, new_steps, filters)
# model.add(layers.MaxPooling1D(5))
# --MaxPooling1D(x) = output only 1/x
# model.add(layers.Flatten())