import gensim

from string import punctuation
from os import listdir
from numpy import array
from numpy import asarray
from numpy import zeros
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import Embedding
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D

# load doc into memory
def load_doc(filename):
    # open the file as read only
    file = open(filename, 'r')
    # read all text
    text = file.read()
    # close the file
    file.close()
    return text

# turn a doc into clean tokens
def clean_doc(doc, vocab):
    # split into tokens by white space
    tokens = doc.split()
    # remove punctuation from each token
    table = str.maketrans('', '', punctuation)
    tokens = [w.translate(table) for w in tokens]
    # filter out tokens not in vocab
    tokens = [w for w in tokens if w in vocab]
    tokens = ' '.join(tokens)
    return tokens

# load all docs in a directory
def process_docs(directory, vocab, is_trian):
    documents = list()
    # walk through all files in the folder
    for filename in listdir(directory):
        # skip any reviews in the test set
        if is_trian and filename.startswith('cv9'):
            continue
        if not is_trian and not filename.startswith('cv9'):
            continue
        # create the full path of the file to open
        path = directory + '/' + filename
        # load the doc
        doc = load_doc(path)
        # clean doc
        tokens = clean_doc(doc, vocab)
        # add to list
        documents.append(tokens)
    return documents

# pad <s> character at the begining
def pad_start_word(docs, value):
    padded_docs = []
    for i in docs:
        padded_docs.append([value] + i)
    return padded_docs

# load the vocabulary
vocab_filename = 'vocab.txt'
vocab = load_doc(vocab_filename)
vocab = vocab.split()
vocab = set(vocab)

# load all training reviews
positive_docs = process_docs('txt_sentoken/pos', vocab, True)
negative_docs = process_docs('txt_sentoken/neg', vocab, True)
train_docs = negative_docs + positive_docs

# create the tokenizer
tokenizer = Tokenizer()
# fit the tokenizer on the documents
tokenizer.fit_on_texts(train_docs)

# define the mapping dictionary for word to id and id to word
word_index = {k:v for k,v in tokenizer.word_index.items()}
word_index['<s>'] = len(tokenizer.word_index) + 1
index_word = {v:k for k,v in word_index.items()}

# define vocabulary size (largest integer value)
vocab_size = len(tokenizer.word_index) + 1 + 1

# sequence encode
encoded_docs = tokenizer.texts_to_sequences(train_docs)
# pad <s> character at the beginning
encoded_docs = pad_start_word(encoded_docs, word_index['<s>'])
# pad sequences
max_length = max([len(s) for s in encoded_docs])
Xtrain = pad_sequences(encoded_docs, maxlen=max_length, padding='pre')
# define training labels
ytrain = array([0 for _ in range(900)] + [1 for _ in range(900)])

# load all test reviews
positive_docs = process_docs('txt_sentoken/pos', vocab, False)
negative_docs = process_docs('txt_sentoken/neg', vocab, False)
test_docs = negative_docs + positive_docs
# sequence encode
encoded_docs = tokenizer.texts_to_sequences(test_docs)
# pad <s> character at the beginning
encoded_docs = pad_start_word(encoded_docs, word_index['<s>'])
# pad sequences
Xtest = pad_sequences(encoded_docs, maxlen=max_length, padding='pre')
# define test labels
ytest = array([0 for _ in range(100)] + [1 for _ in range(100)])

# define word2vec embedding layer
word2vec_model = gensim.models.Word2Vec.load("../../word2vec/wiki.en.200.word2vec")
embedding_layer = word2vec_model.wv.get_keras_embedding(train_embeddings=False)
embedding_layer.input_length = max_length

# # define CNN model
# model = Sequential()
# model.add(embedding_layer)
# model.add(Conv1D(filters=32, kernel_size=8, activation='relu'))
# model.add(MaxPooling1D(pool_size=2))
# # model.add(Flatten())
# model.add(LSTM(64))
# model.add(Dense(10, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(1, activation='sigmoid'))
# print(model.summary())
# # compile network
# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# # fit network
# model.fit(Xtrain, ytrain, validation_data=(Xtest, ytest), epochs=50, verbose=2)
# # evaluate
# loss, acc = model.evaluate(Xtest, ytest, verbose=0)
# print('Test Accuracy: %f' % (acc*100))

# define LSTM model
model = Sequential()
model.add(embedding_layer)
model.add(LSTM(64))
model.add(Dense(1, activation='sigmoid'))
print(model.summary())
# compile network
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# fit network
model.fit(Xtrain, ytrain, validation_data=(Xtest, ytest), epochs=10, verbose=2)
# evaluate
loss, acc = model.evaluate(Xtest, ytest, verbose=0)
print('Test Accuracy: %f' % (acc*100))


