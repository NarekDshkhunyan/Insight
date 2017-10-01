'''
This script loads pre-trained word embeddings (GloVe embeddings) into a frozen Keras Embedding layer,
and uses it to train a text classification model with a ConvNet architecture

'''
from keras.layers import Dense, Input, Embedding, Dropout
from keras.layers import Conv1D, MaxPooling1D, Flatten
from keras.layers import SimpleRNN, LSTM, GRU, Bidirectional
from keras.models import Model


# -----------------------------------------------------------------------------------------------------------
def LSTMNet(embeddings, MAX_SEQUENCE_LENGTH, num_words, EMBEDDING_DIM, labels_index, CNN = False, train = False):

    # load pre-trained word embeddings into an Embedding layer
    # note that we set trainable = False so as to keep the embeddings fixed
    embedding_layer = Embedding(input_dim=num_words,
                                output_dim=EMBEDDING_DIM,
                                weights=[embeddings],
                                input_length=MAX_SEQUENCE_LENGTH,
                                trainable=train)

    print('Training model.')

    # get the input for the LSTM network
    sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    embedded_sequences = embedding_layer(sequence_input)

    if not CNN:
        lstm = Bidirectional(LSTM(100, return_sequences=False))(embedded_sequences)    # put return_sequences=True if running CNN on top of LSTM
        output = Dropout(0.5)(lstm)
    else:
        lstm = Bidirectional(LSTM(100, return_sequences=True))(embedded_sequences)
        lstm = Dropout(0.5)(lstm)
        # add a 1D convnet with global maxpooling
        conv_1 = Conv1D(filters=128, kernel_size=3, activation='relu')(lstm)
        pool_1 = MaxPooling1D(pool_size=3)(conv_1)
        x = Dropout(0.5)(pool_1)
        x = Flatten()(x)                           # uncomment if no LSTM
        output = Dense(128, activation='relu')(x)
        #output = Dropout(0.5)(output)

    preds = Dense(labels_index, activation='softmax')(output)                          # use LSTM instead of x if not using CNN

    # print sequence_input.shape
    # print embedded_sequences.shape
    # print lstm.shape
    # print conv_1.shape, pool_1.shape
    # print output.shape
    # print preds.shape

    model = Model(sequence_input, preds)
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    return model
