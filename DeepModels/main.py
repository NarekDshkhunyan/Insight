import cPickle

from modelCNN import ConvNet
from modelLSTM import LSTMNet
from modelAttention import AttentionNet

from sklearn.metrics import confusion_matrix
from vis import plot_confusion_matrix
import numpy as np


# -------------------------------------------------------------------------------------------------------------
MAX_SEQUENCE_LENGTH = 20
num_words = 1880               # number of unique words 1724
EMBEDDING_DIM = 300
labels_index = 33              # number of labels       20
CNN = False                    # whether to build a ConvNet on top of the LSTM
train = False                  # whether to train the embeddings as well

# -------------------------------------------------------------------------------------------------------------
# load the relevant pickles
train_matrix_file = "Data/train_matrix.pkl"
val_matrix_file = "Data/val_matrix.pkl"
embedding_matrix_file = "Data/embedding_matrix.pkl"
with open(train_matrix_file) as f:
    x_train, y_train = cPickle.load(f)
with open(val_matrix_file) as f:
    x_val, y_val = cPickle.load(f)
with open(embedding_matrix_file) as f:
    embedding_matrix = cPickle.load(f)

# -------------------------------------------------------------------------------------------------------------
#model = ConvNet(embedding_matrix, MAX_SEQUENCE_LENGTH, num_words, EMBEDDING_DIM, labels_index, train)
#model = LSTMNet(embedding_matrix, MAX_SEQUENCE_LENGTH, num_words, EMBEDDING_DIM, labels_index)
model = AttentionNet(embedding_matrix, MAX_SEQUENCE_LENGTH, num_words, EMBEDDING_DIM, labels_index)
#model.summary()

model.fit(x_train, y_train,
          batch_size=32,
          epochs=10,
          validation_data=(x_val, y_val))

score, acc = model.evaluate(x_val, y_val,
                            batch_size=32)
print('Test score:', score)
print('Test accuracy:', acc)

y_pred = model.predict(x_val)
true = np.argmax(y_val, axis=1)
pred = np.argmax(y_pred, axis=1)
cm = confusion_matrix(true, pred)
plot = plot_confusion_matrix(cm, np.arange(33), normalize=True, title='Confusion Matrix')
plot.show()
