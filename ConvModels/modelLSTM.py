'''This script loads pre-trained word embeddings (GloVe embeddings) into a frozen Keras Embedding layer,
and uses it to train a text classification model on the 20 Newsgroup dataset
(classification of newsgroup messages into 20 different categories).

GloVe embedding data can be found at:
http://nlp.stanford.edu/data/glove.6B.zip
(source page: http://nlp.stanford.edu/projects/glove/)

20 Newsgroup data can be found at:
http://www.cs.cmu.edu/afs/cs.cmu.edu/project/theo-20/www/data/news20.html
'''

import cPickle

from keras.layers import Dense, Input, Embedding, Dropout
from keras.layers import Conv1D, MaxPooling1D, Flatten
from keras.layers import LSTM, GRU, Bidirectional
from keras.models import Model


# -------------------------------------------------------------------------------------------------------------
MAX_SEQUENCE_LENGTH = 20
num_words = 1880               # number of unique words 1724
EMBEDDING_DIM = 300
labels_index = 33              # number of labels       20

# load the relevant pickles
embedding_matrix_file = "Data/embedding_matrix.pkl"
train_matrix_file = "Data/train_matrix.pkl"
val_matrix_file = "Data/val_matrix.pkl"
with open(embedding_matrix_file) as f:
    embedding_matrix = cPickle.load(f)
with open(train_matrix_file) as f:
    x_train, y_train = cPickle.load(f)
with open(val_matrix_file) as f:
    x_val, y_val = cPickle.load(f)
# -----------------------------------------------------------------------------------------------------------


# load pre-trained word embeddings into an Embedding layer
# note that we set trainable = False so as to keep the embeddings fixed
embedding_layer = Embedding(input_dim=num_words,
                            output_dim=EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=False)

print('Training model.')

# train an LSTM for classification
sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
embedded_sequences = embedding_layer(sequence_input)
lstm = Bidirectional(LSTM(100, return_sequences=True))(embedded_sequences)    # put return_sequences=True if running CNN on top of LSTM
lstm = Dropout(0.5)(lstm)

# add a 1D convnet with global maxpooling
conv_1 = Conv1D(filters=128, kernel_size=3, activation='relu')(lstm)
pool_1 = MaxPooling1D(pool_size=3)(conv_1)
x = Dropout(0.5)(pool_1)
x = Flatten()(x)                           # uncomment if no LSTM
x = Dense(128, activation='relu')(x)
#x = Dropout(0.5)(x)

preds = Dense(labels_index, activation='softmax')(x)                          # use LSTM instead of x if not using CNN

# print sequence_input.shape
# print embedded_sequences.shape
# print lstm.shape
# print conv_1.shape, pool_1.shape
# print x.shape
# print preds.shape

model = Model(sequence_input, preds)
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
#model.summary()

model.fit(x_train, y_train,
          batch_size=32,
          epochs=10,
          validation_data=(x_val, y_val))

score, acc = model.evaluate(x_val, y_val,
                            batch_size=32)
print('Test score:', score)
print('Test accuracy:', acc)
