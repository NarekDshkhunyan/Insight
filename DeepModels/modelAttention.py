'''
This script loads pre-trained word embeddings (GloVe embeddings) into a frozen Keras Embedding layer,
and uses it to train a text classification model with a ConvNet architecture

'''
from keras.layers import Dense, Input, Embedding, Dropout
from keras.layers import LSTM, Bidirectional, GRU
from keras.models import Model

from keras import backend as K
from keras.layers import Lambda
from keras.engine.topology import Layer, InputSpec
from keras import initializers

from keras.layers import Flatten, merge, TimeDistributed, Activation, RepeatVector, Permute

# -------------------------------------------------------------------------------------------------------------
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


# -------------------------------------------------------------------------------------------------------------
def AttentionNet(embeddings, MAX_SEQUENCE_LENGTH, num_words, EMBEDDING_DIM, labels_index, train = False):

    # load pre-trained word embeddings into an Embedding layer
    # note that we set trainable = False so as to keep the embeddings fixed
    embedding_layer = Embedding(num_words,
                                EMBEDDING_DIM,
                                weights=[embeddings],
                                input_length=MAX_SEQUENCE_LENGTH,
                                trainable=train)

    print('Training model.')

    # get the input for the LSTM network
    sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    embedded_sequences = embedding_layer(sequence_input)

    lstm = Bidirectional(LSTM(100, return_sequences=True))(embedded_sequences)

    # compute importance for each step
    attention = Dense(1, activation='tanh')(lstm)
    attention = Flatten()(attention)
    attention = Activation('softmax')(attention)
    attention = RepeatVector(200)(attention)
    attention = Permute([2, 1])(attention)

    #print lstm.shape
    #print attention.shape

    # apply the attention
    sent_representation = merge([lstm, attention], mode='mul')
    sent_representation = Lambda(lambda xin: K.sum(xin, axis=1))(sent_representation)

    # gru = Bidirectional(GRU(100, return_sequences=True))(embedded_sequences)
    # att = AttLayer()(gru)
    # x = Dense(128, activation='relu')(att)
    #lstm = Dropout(0.5)(lstm)

    preds = Dense(labels_index, activation='softmax')(sent_representation)
    print preds.shape

    model = Model(sequence_input, preds)
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    return model