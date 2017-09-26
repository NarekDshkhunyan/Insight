import os, cPickle
import numpy as np

BASE_DIR = '/home/narek/Dropbox/Insight/IMDB'
GLOVE_DIR = BASE_DIR + '/glove.6B/'

MAX_NB_WORDS = 2000             # 20000
EMBEDDING_DIM = 300

# load the word index dictionary
wordindex_file = "Data/word_index.pkl"
with open(wordindex_file) as f:
    word_index = cPickle.load(f)

# build index mapping words in the embeddings set to their embedding vector
print('Indexing word vectors.')

embeddings_index = {}
f = open(os.path.join(GLOVE_DIR, 'glove.6B.300d.txt'))
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Found %s word vectors.' % len(embeddings_index))

print('Preparing embedding matrix.')

# prepare embedding matrix
num_words = min(MAX_NB_WORDS, len(word_index))
embedding_matrix = np.zeros((num_words + 1, EMBEDDING_DIM))
for word, i in word_index.items():
    if i >= MAX_NB_WORDS:
        continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector


cPickle.dump(embedding_matrix, open('Data/embedding_matrix.pkl', 'wb'))