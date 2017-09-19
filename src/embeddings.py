from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from collections import defaultdict
import numpy as np
import cPickle
from utils import load_data


NUM_FEATURES = 19

data_file = "../Data/word2vec.pkl"
with open(data_file) as f:
    word2vec = cPickle.load(f)

k = len(word2vec.itervalues().next())


# -------------------------------------------------------------------------------------------------------------
def cv(train_data, test_data):
    count_vectorizer = CountVectorizer()

    train = count_vectorizer.fit_transform(train_data)
    test = count_vectorizer.transform(test_data)
    features = count_vectorizer.get_feature_names()

    return train, test, count_vectorizer


def tfidf(train_data, test_data):
    tfidf_vectorizer = TfidfVectorizer()

    train = tfidf_vectorizer.fit_transform(train_data)
    test = tfidf_vectorizer.transform(test_data)
    features = tfidf_vectorizer.get_feature_names()

    return train, test, tfidf_vectorizer


def avg_feature_vector(sentences, model, NUM_FEATURES, k = 300):
    """ Concatenates all words vector in a given sentence """

    featureVec = np.zeros((len(sentences), NUM_FEATURES * k), dtype="float32")
    for i in xrange(len(sentences)):
        for j in xrange(len(sentences[i])):
            sentence = sentences[i]
            word = sentence[j]
            if word in model:
                featureVec[i][300*j:300*(j+1)] = model[word]
            else:
                #print word
                featureVec[i][300*j:300*(j+1)] = np.random.uniform(-0.25, 0.25, k)
        #featureVec[i] = np.divide(featureVec[i], len(sentences[i]))

    return featureVec


def mean_embedding_vectorizer(sentences, model, tfidf_vec):
    """ Averages all word vectors in a given sentence """

    max_idf = max(tfidf_vec.idf_)
    word2weight = defaultdict(lambda: max_idf, [(w, tfidf_vec.idf_[i]) for w, i in tfidf_vec.vocabulary_.items()])
    return np.array([np.mean([model[w]*word2weight[w] for w in words if w in model] or [np.zeros(300)], axis=0) for words in sentences])


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

    if type == 'mean_emb':
        return train_embeddings, test_embeddings
    if type == 'concat_emb':
        return
    if type == 'cv':
        return train_cv, test_cv
    if type == 'tfidf':
        return train_tf, test_tf

def avg_feature_vector(sentences, model, k):
    """ Concatenates all words vector in a given sentence """

    train = np.stack([np.stack([model[word] for word in sentence if word in model] or [np.zeros(k)]) for sentence in sentences])
    return train
    #return np.array([np.concatenate([model[w] for w in words if w in model] or [np.zeros(k)], axis=0) for words in sentences])
