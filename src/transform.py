import cPickle
from embeddings import *
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

from utils import *

# -------------------------------------------------------------------------------------------------------------
def produceEmbeddings(type, train_data, test_data):

    train_sentences = load_data(train_data)
    test_sentences = load_data(test_data)

    train_tf, test_tf, tfidf_vectorizer = tfidf(train_data, test_data)
    train_cv, test_cv, count_vectorizer = cv(train_data, test_data)

    #train_embeddings = avg_feature_vector(train_sentences, word2vec, NUM_FEATURES, train, tfidf_vectorizer, 300)
    #test_embeddings = avg_feature_vector(test_sentences, word2vec, NUM_FEATURES, test, tfidf_vectorizer, 300)

    train_embeddings = mean_embedding_vectorizer(train_sentences, word2vec, tfidf_vectorizer)
    test_embeddings = mean_embedding_vectorizer(test_sentences, word2vec, tfidf_vectorizer)

    if type == 'mean_emb': return train_embeddings, test_embeddings
    if type == 'concat_emb': return
    if type == 'cv': return train_cv, test_cv
    if type == 'tfidf': return train_tf, test_tf

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
X_train, X_test = produceEmbeddings('mean_emb', X_train, X_test)
cPickle.dump([X_train, X_test, y_train, y_test], open('../Data/mean_embeddings.pkl', 'wb'))
