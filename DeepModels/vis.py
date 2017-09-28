import cPickle

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

embedding_matrix_file = "Data/embedding_matrix.pkl"
with open(embedding_matrix_file) as f:
    embedding_matrix = cPickle.load(f)

pca = PCA(n_components=2)
x_pca = pca.fit_transform(embedding_matrix)
plt.figure(1, figsize=(30, 20),)
plt.scatter(x_pca[:, 0], x_pca[:, 1],s=100, c=y, alpha=0.2)

X_tsne = TSNE(n_components=2, verbose=2).fit_transform(embedding_matrix)
plt.figure(1, figsize=(30, 20),)
plt.scatter(X_tsne[:, 0], X_tsne[:, 1],s=100, c=y, alpha=0.2)
