import cPickle
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

from sklearn.metrics import confusion_matrix
from confusion_matrix import plot_confusion_matrix

from transform import *
from classify import *

import argparse

# -------------------------------------------------------------------------------------------------------------
data_file = "../Data/train_mat_filtered.pkl"
#vocab_file = "../Data/vocab_filtered.pkl"
vocab_inv_file = "../Data/vocab_inv_filtered.pkl"

with open(data_file) as f:
    data, labels, _ = cPickle.load(f)
with open(vocab_inv_file) as f:
    vocabulary_inv = cPickle.load(f)

# -------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Classify the given sentences.')
    parser.add_argument('algorithm', default='lr',
                        help='the linear classifier to run')
    parser.add_argument('embedding', default='tfidf',
                        help='which embedding to use')
    parser.add_argument('plot_confusion_matrix', default='False',
                        help='whether to plot the confusion matrix')

    args = parser.parse_args()
    print args

    embeddings, labels = getEmbeddings(data, labels, vocabulary_inv, args.embedding)
    # Random split
    X_train, X_test, y_train, y_test = train_test_split(embeddings, labels, test_size=0.2, random_state=42)
    print X_train.shape, X_test.shape
    #cPickle.dump([X_train, X_test, y_train, y_test], open('../Data/mean_embeddings.pkl', 'wb'))

    # Run the classification algorithm
    results = {'accuracy': [], 'precision': [], 'recall': [], 'f1': []}
    y_predicted = classify(args.algorithm, X_train, y_train, X_test, y_test, results)
    cm = confusion_matrix(y_test, y_predicted)

    # Cross-validation
    # for k, (train, test) in enumerate(KFold(3).split(embeddings, labels)):
    #    X_train, X_test, y_train, y_test = embeddings[train], embeddings[test], labels[train], labels[test]
    #    print classify(algorithm, X_train, y_train, X_test, y_test, results)

    if args.plot_confusion_matrix == 'True':
        plot = plot_confusion_matrix(cm, classes=np.arange(33), normalize=True, title='Confusion matrix')
        plot.show()