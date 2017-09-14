from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import numpy as np
import cPickle
from preprocess import load_data


NUM_FEATURES = 19

data_file = "../Data/word2vec.pkl"
with open(data_file) as f:
    word2vec = cPickle.load(f)


# -------------------------------------------------------------------------------------------------------------
def cv(train_data, test_data):
    count_vectorizer = CountVectorizer()

    train = count_vectorizer.fit_transform(train_data)
    test = count_vectorizer.transform(test_data)
    features = count_vectorizer.get_feature_names()

    return train, test, features


def tfidf(train_data, test_data):
    tfidf_vectorizer = TfidfVectorizer()

    train = tfidf_vectorizer.fit_transform(train_data)
    test = tfidf_vectorizer.transform(test_data)
    features = tfidf_vectorizer.get_feature_names()

    return train, test, tfidf_vectorizer


def avg_feature_vector(sentences, model, NUM_FEATURES, tfidf_embeddings, tfidf_vec, k = 300):
    # function to average all words vectors in a given paragraph
    featureVec = np.zeros((len(sentences), NUM_FEATURES * k), dtype="float32")
    for i in xrange(len(sentences)):
        for j in xrange(len(sentences[i])):
            sentence = sentences[i]
            word = sentence[j]
            if word in model and word in tfidf_vec.vocabulary_:
                a = tfidf_vec.vocabulary_[word]
                #print a, word
                featureVec[i][300*j:300*(j+1)] = model[word] * tfidf_embeddings[j, a]
            else:
                print word
                featureVec[i][300*j:300*(j+1)] = np.random.uniform(-0.25, 0.25, k)
                #featureVec[sentence][word] = model[word] * train_embeddings[word]
        featureVec[i] = np.divide(featureVec[i], len(sentences[i]))

    return featureVec

train, test, tfidf_vectorizer = tfidf(train_data, test_data)
# -------------------------------------------------------------------------------------------------------------
def produceEmbeddings(train_data, test_data):

    train_sentences = load_data(train_data)
    test_sentences = load_data(test_data)
    train_embeddings = avg_feature_vector(train_sentences, word2vec, NUM_FEATURES, train, tfidf_vectorizer, 300)
    test_embeddings = avg_feature_vector(test_sentences, word2vec, NUM_FEATURES, test, tfidf_vectorizer, 300)

    return train_embeddings, test_embeddings
