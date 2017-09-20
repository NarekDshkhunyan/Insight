import cPickle
from embeddings import *
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

from utils import *

# -------------------------------------------------------------------------------------------------------------
def produceEmbeddings(type, data):

    sentences = pad_sentences(load_data(data))

    tf_emb, tfidf_vectorizer = tfidf(data)
    cv_emb, count_vectorizer = cv(data)

    #concat_emb = avg_feature_vector(sentences, 300)

    mean_emb = mean_embedding_vectorizer(sentences, word2vec, tfidf_vectorizer)

    if type == 'mean_emb': return mean_emb
    #if type == 'concat_emb': return concat_emb
    if type == 'cv': return cv_emb
    if type == 'tfidf': return tf_emb

# -------------------------------------------------------------------------------------------------------------
def getEmb(data, labels, vocabulary_inv):
    data = clean_data(transform_data(vocabulary_inv, data))
    questions, labels = get_random_samples(data, labels)

    # Get embeddings
    embeddings = produceEmbeddings('mean_emb', questions)
    return embeddings

# Random split - think about KFold as well
X_train, X_test, y_train, y_test = train_test_split(embeddings, labels, test_size=0.1, random_state=42)

# Save the mebeddings
cPickle.dump([X_train, X_test, y_train, y_test], open('../Data/mean_embeddings.pkl', 'wb'))
