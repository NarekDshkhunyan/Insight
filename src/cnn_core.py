import numpy as np
import cPickle

from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Input, MaxPooling1D, Convolution1D, Embedding, Convolution2D, MaxPooling2D
from keras.layers.merge import Concatenate
from keras.preprocessing import sequence

from sklearn.metrics import log_loss
np.random.seed(0)

# ---------------------- Parameters section -------------------
# Model Hyperparameters
embedding_dim = 300
filter_sizes = (3, 8)
num_filters = 10
dropout_prob = (0.5, 0.8)
hidden_dims = 50

# Training parameters
batch_size = 16
num_epochs = 10

# Prepossessing parameters
sequence_length = 19
max_words = 5000

# ---------------------- Parameters end -----------------------
print("Load data...")

input_shape = (sequence_length, embedding_dim)   # sequence_length

def CNN():
    model_input = Input(shape=input_shape)
    z = model_input
    z = Dropout(dropout_prob[0])(z)

    # Convolutional block
    conv_blocks = []
    for sz in filter_sizes:
        conv = Convolution1D(filters=num_filters,
                         kernel_size=sz,
                         padding="same",
                         activation="relu",
                         strides=1)(z)
        conv = MaxPooling1D(pool_size=2)(conv)
        conv = Flatten()(conv)
        conv_blocks.append(conv)
    z = Concatenate()(conv_blocks) if len(conv_blocks) > 1 else conv_blocks[0]

    z = Dropout(dropout_prob[1])(z)
    z = Dense(hidden_dims, activation="relu")(z)
    model_output = Dense(1, activation="sigmoid")(z)

    model = Model(model_input, model_output)
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

    return model