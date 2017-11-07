# LSTM and CNN for sequence classification in the IMDB dataset
import numpy
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers.embeddings import Embedding
from keras.preprocessing.sequence import pad_sequences

# pad <s> character at the begining
def pad_start_word(docs, value):
    padded_docs = []
    for i in docs:
        padded_docs.append([value] + i)
    return padded_docs

# classify a review as negative (0) or positive (1)
def predict_sentiment(review, word_index, max_sequences_length, model):
    tokens = review.split()
    # convert to word index list
    line = [word_index[i] for i in tokens]
    # pad start word and 0
    line = pad_start_word([line], word_index['<s>'])
    line = pad_sequences(line, maxlen=max_sequences_length, padding='pre')
    # prediction
    # yhat = model.predict_classes(line, verbose=0)
    yhat = model.predict(line)
    return yhat[0,0]

# fix random seed for reproducibility
numpy.random.seed(7)

# load the dataset but only keep the top n words, zero the rest
top_words = 10000
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=top_words)

# create the mapping dictionary
word_index = {k:(v+3) for k,v in imdb.get_word_index().items()}
word_index['<s>'] = 1
word_index['<PAD>'] = 0
word_index['<UNK>'] = 2
index_word = {v:k for k,v in word_index.items()}

# pad input sequences
max_review_length = max([len(i) for i in X_train])
X_train = pad_sequences(X_train, maxlen=max_review_length, padding='pre')
X_test = pad_sequences(X_test, maxlen=max_review_length, padding='pre')

# create the LSTM-CNN model
model = Sequential()
model.add(Embedding(top_words, 100, input_length=max_review_length))
model.add(Conv1D(filters=32, kernel_size=8, activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(LSTM(100))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
model.fit(X_train, y_train, validation_data=(X_test,y_test), epochs=10, batch_size=64, verbose=2)

# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))