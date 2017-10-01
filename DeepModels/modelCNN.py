'''
This script loads pre-trained word embeddings (GloVe embeddings) into a frozen Keras Embedding layer,
and uses it to train a text classification model with a ConvNet architecture

'''

from keras.layers import Dense, Input, Flatten, Dropout, Merge
from keras.layers import Conv1D, MaxPooling1D, Embedding
from keras.layers import LSTM, Bidirectional
from keras.models import Model

# -------------------------------------------------------------------------------------------------------------
def ConvNet(embeddings, MAX_SEQUENCE_LENGTH, num_words, EMBEDDING_DIM, labels_index):

    # load pre-trained word embeddings into an Embedding layer
    # note that we set trainable = False so as to keep the embeddings fixed
    embedding_layer = Embedding(num_words,
                            EMBEDDING_DIM,
                            weights=[embeddings],
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=True)

    print('Training model.')

    # get the input for the CNN network
    sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    embedded_sequences = embedding_layer(sequence_input)

    # -------------------------------------------------------------------------------------------------------------
    # Yoon Kim model
    convs = []
    filter_sizes = [3,4,5]

    for fsz in filter_sizes:
        l_conv = Conv1D(filters=128, kernel_size=fsz, activation='relu')(embedded_sequences)
        l_pool = MaxPooling1D(pool_size=3)(l_conv)
        convs.append(l_pool)

    l_merge = Merge(mode='concat', concat_axis=1)(convs)
    # -------------------------------------------------------------------------------------------------------------

    # add a 1D convnet with global maxpooling, instead of Yoon Kim model
    conv = Conv1D(filters=128, kernel_size=3, activation='relu')(embedded_sequences)
    pool = MaxPooling1D(pool_size=3)(conv)

    #lstm = Bidirectional(LSTM(100, return_sequences=False))(pool_1)
    x = Dropout(0.5)(l_merge)                  # use 'l_merge' for Yoon Kim or 'pool' for 1D ConvNet
    x = Flatten()(x)                           # uncomment if no LSTM
    x = Dense(128, activation='relu')(x)
    #x = Dropout(0.5)(x)

    preds = Dense(labels_index, activation='softmax')(x)

    model = Model(sequence_input, preds)
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['acc'])

    return model