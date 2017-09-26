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

from keras.layers import Dense, Input, Embedding
from keras.layers import LSTM, Bidirectional, GRU
from keras.models import Model

from keras import backend as K
from keras.engine.topology import Layer, InputSpec
from keras import initializers

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


class AttLayer(Layer):
    def __init__(self, **kwargs):
        self.init = initializers.get('normal')
        #self.input_spec = [InputSpec(ndim=3)]
        super(AttLayer, self).__init__(** kwargs)

    def build(self, input_shape):
        assert len(input_shape)==3
        #self.W = self.init((input_shape[-1],1))
        self.W = self.init((input_shape[-1],))
        #self.input_spec = [InputSpec(shape=input_shape)]
        self.trainable_weights = [self.W]
        super(AttLayer, self).build(input_shape)  # be sure you call this somewhere!

    def call(self, x, mask=None):
        eij = K.tanh(K.dot(x, self.W))

        ai = K.exp(eij)
        weights = ai/K.sum(ai, axis=1).dimshuffle(0,'x')

        weighted_input = x*weights.dimshuffle(0,1,'x')
        return weighted_input.sum(axis=1)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])


# load pre-trained word embeddings into an Embedding layer
# note that we set trainable = False so as to keep the embeddings fixed
embedding_layer = Embedding(num_words,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=False)

print('Training model.')

# train a 1D convnet with global maxpooling
sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
embedded_sequences = embedding_layer(sequence_input)
l_lstm = Bidirectional(LSTM(100))(embedded_sequences)
l_gru = Bidirectional(GRU(100, return_sequences=True))(embedded_sequences)
l_att = AttLayer()(l_gru)

#x = Dense(128, activation='relu')(l_att)
#x = Dropout(0.5)(x)

preds = Dense(labels_index, activation='softmax')(l_att)

model = Model(sequence_input, preds)
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['acc'])
#model.summary()

model.fit(x_train, y_train,
          batch_size=32,
          epochs=10,
          validation_data=(x_val, y_val))

score, acc = model.evaluate(x_val, y_val,
                            batch_size=32)
print('Test score:', score)
print('Test accuracy:', acc)
