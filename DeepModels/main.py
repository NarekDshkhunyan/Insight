import cPickle

from modelCNN import ConvNet
from modelLSTM import LSTMNet
from modelAttention import AttentionNet

from sklearn.metrics import confusion_matrix
from vis import plot_confusion_matrix
import numpy as np

import argparse

# -------------------------------------------------------------------------------------------------------------
MAX_SEQUENCE_LENGTH = 19
num_words = 1880              # number of unique words 1996
EMBEDDING_DIM = 300
labels_index = 33              # number of labels       58
CNN = False                    # whether to build a ConvNet on top of the LSTM
train = True                   # whether to train the embeddings as well

# -------------------------------------------------------------------------------------------------------------
# load the relevant pickles
train_matrix_file = "../Data/train_matrix.pkl"
val_matrix_file = "../Data/val_matrix.pkl"
embedding_matrix_file = "../Data/embedding_matrix.pkl"
with open(train_matrix_file) as f:
    x_train, y_train = cPickle.load(f)
with open(val_matrix_file) as f:
    x_val, y_val = cPickle.load(f)
with open(embedding_matrix_file) as f:
    embedding_matrix = cPickle.load(f)

print x_train.shape, x_val.shape, embedding_matrix.shape

# -------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Classify the given sentences.')
    parser.add_argument('algorithm', default='ConvNet',
                        help='the linear classifier to run')
    # parser.add_argument('train_embeddings', type=bool, default=False,
    #                     help='Whether to train the embedding weights as well')
    parser.add_argument('plot_confusion_matrix', default='False',
                        help='Whether to plot the confusion matrix')

    args = parser.parse_args()
    print args

    if args.algorithm == 'ConvNet':
        model = ConvNet(embedding_matrix, MAX_SEQUENCE_LENGTH, num_words, EMBEDDING_DIM, labels_index, train)
    elif args.algorithm == 'LSTM':
        model = LSTMNet(embedding_matrix, MAX_SEQUENCE_LENGTH, num_words, EMBEDDING_DIM, labels_index, train)
    elif args.algorithm == 'Attention':
        model = AttentionNet(embedding_matrix, MAX_SEQUENCE_LENGTH, num_words, EMBEDDING_DIM, labels_index, train)

    #model.summary()

    model.fit(x_train, y_train,
              batch_size=32,
              epochs=10,
            validation_data=(x_val, y_val),
            class_weight='auto')

    score, acc = model.evaluate(x_val, y_val,
                            batch_size=32)
    print('Test score:', score)
    print('Test accuracy:', acc)

    y_pred = model.predict(x_val)
    true = np.argmax(y_val, axis=1)
    pred = np.argmax(y_pred, axis=1)

    if args.plot_confusion_matrix == 'True':
        cm = confusion_matrix(true, pred)
        plot = plot_confusion_matrix(cm, np.arange(33), normalize=True, title='Confusion Matrix')
        plot.show()
