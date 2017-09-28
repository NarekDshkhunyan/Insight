import numpy as np
import cPickle
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from collections import defaultdict


NUM_FEATURES = 19

data_file = "../Data/word2vec.pkl"
with open(data_file) as f:
    word2vec = cPickle.load(f)

k = len(word2vec.itervalues().next())


# -------------------------------------------------------------------------------------------------------------
def cv(data):
    count_vectorizer = CountVectorizer(ngram_range=(1,2))

    emb = count_vectorizer.fit_transform(data)

    return emb, count_vectorizer


def tfidf(data):
    tfidf_vectorizer = TfidfVectorizer(ngram_range=(1,2))

    train = tfidf_vectorizer.fit_transform(data)

    return train, tfidf_vectorizer


def mean_embedding_vectorizer(sentences, model, tfidf_vec):
    """ Averages all word vectors in a given sentence """

    max_idf = max(tfidf_vec.idf_)
    word2weight = defaultdict(lambda: max_idf, [(w, tfidf_vec.idf_[i]) for w, i in tfidf_vec.vocabulary_.items()])
    return np.array([np.mean([model[w]*word2weight[w] for w in words if w in model] or [np.zeros(300)], axis=0) for words in sentences]) # *word2weight[w]
    #return np.stack([np.mean([model[word] for word in sentence if word in model] or [np.zeros(k)]) for sentence in sentences])


def avg_feature_vector(sentences, model, k):
    """ Concatenates all words vector in a given sentence """

    return np.stack([np.stack([model[word] if word in model else np.zeros(k) for word in sentence]) for sentence in sentences]) # np.random.uniform(-0.25, 0.25, k)
    #return np.array([np.concatenate([model[w] for w in words if w in model] or [np.zeros(k)], axis=0) for words in sentences])
