import cPickle
from embeddings import produceEmbeddings
from sklearn.model_selection import train_test_split

from utils import *


# -------------------------------------------------------------------------------------------------------------
data_file = "../Data/train_mat_filtered.pkl"
#vocab_file = "../Data/vocab_filtered.pkl"
vocab_inv_file = "../Data/vocab_inv_filtered.pkl"

with open(data_file) as f:
    data, labels, _ = cPickle.load(f)
with open(vocab_inv_file) as f:
    vocabulary_inv = cPickle.load(f)

data = clean_data(transform_data(vocabulary_inv, data))
questions, labels = get_random_samples(data, labels)
# Random split - think about KFold as well
X_train, X_test, y_train, y_test = train_test_split(questions, labels, test_size=0.1, random_state=42)

# Get embeddings
X_train, X_test = produceEmbeddings('tfidf', X_train, X_test)
cPickle.dump([X_train, X_test, y_train, y_test], open('../Data/tfidf_embeddings.pkl', 'wb'))
