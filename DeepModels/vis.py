import cPickle

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# embedding_matrix_file = "../Data/embedding_matrix.pkl"
# with open(embedding_matrix_file) as f:
#     embedding_matrix = cPickle.load(f)
#
# pca = PCA(n_components=2)
# x_pca = pca.fit_transform(embedding_matrix)
# plt.figure(1, figsize=(30, 20),)
# plt.scatter(x_pca[:, 0], x_pca[:, 1],s=100, alpha=0.2)

#X_tsne = TSNE(n_components=2, verbose=2).fit_transform(embedding_matrix)
#plt.figure(1, figsize=(30, 20),)
#plt.scatter(X_tsne[:, 0], X_tsne[:, 1],s=100, c=y, alpha=0.2)


import numpy as np
import itertools

#-------------------------------------------------------------------------------------------------------------
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.1f' if normalize else 'd'
    thresh = cm.max() / 2.
    # for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    #     plt.text(j, i, format(cm[i, j], fmt),
    #              horizontalalignment="center",
    #              color="white" if cm[i, j] > thresh else "black")

    #plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    return plt
