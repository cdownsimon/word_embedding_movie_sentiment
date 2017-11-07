from string import punctuation
from os import listdir
from numpy import array
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
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

# load doc and add to vocab
def add_doc_to_vocab(filename, vocab):
    # load doc
    doc = load_doc(filename)
    # clean doc
    tokens = clean_doc(doc)
    # update counts
    vocab.update(tokens)

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
        
# save list to file
def save_list(lines, filename):
    # convert lines to a single blob of text
    data = '\n'.join(lines)
    # open file
    file = open(filename, 'w')
    # write text
    file.write(data)
    # close file
    file.close()

# classify a review as negative (0) or positive (1)
def predict_sentiment(review, vocab, word_index, max_sequences_length, model):
    # clean
    tokens = clean_doc(review, vocab)
    tokens = tokens.split()
    # convert to word index list
    line = [word_index[i] for i in tokens]
    # pad start word and 0
    line = pad_start_word([line], word_index['<s>'])
    line = pad_sequences(line, maxlen=max_sequences_length, padding='pre')
    # prediction
    yhat = model.predict_classes(line, verbose=0)
    return yhat[0,0]

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

# # define CNN model
# print('Building CNN model...')
# model = Sequential()
# model.add(Embedding(vocab_size, 100, input_length=max_length))
# model.add(Conv1D(filters=32, kernel_size=8, activation='relu'))
# model.add(MaxPooling1D(pool_size=2))
# model.add(Flatten())
# model.add(Dense(10, activation='relu'))
# model.add(Dense(1, activation='sigmoid'))
# print(model.summary())
# 
# # compile network
# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# # fit network
# model.fit(Xtrain, ytrain, validation_data=(Xtest, ytest), epochs=10, verbose=2)
# 
# # evaluate
# loss, acc = model.evaluate(Xtest, ytest, verbose=0)
# print('Test Accuracy: %f' % (acc*100))

# define LSTM model
print('Building LSTM model...')
model = Sequential()
model.add(Embedding(vocab_size, 100, input_length=max_length))
model.add(LSTM(64, dropout = 0.5, recurrent_dropout = 0.5))
model.add(Dense(1, activation='sigmoid'))
print(model.summary())

# compile network
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# fit network
model.fit(Xtrain, ytrain,  validation_data=(Xtest, ytest), epochs=30, verbose=2)

# evaluate
loss, acc = model.evaluate(Xtest, ytest, verbose=0)
print('Test Accuracy: %f\n' % (acc*100))


