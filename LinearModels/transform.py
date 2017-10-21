from embeddings import *
from utils import *

# -------------------------------------------------------------------------------------------------------------
def produceEmbeddings(type, data):
    """ Produce embeddings of the choice """

    sentences = pad_sentences(load_data(data))

    tf_emb, tfidf_vectorizer = tfidf(data)
    cv_emb, count_vectorizer = cv(data)

    concat_emb = concat_feature_vector(sentences, word2vec, 300)

    mean_emb = mean_embedding_vectorizer(sentences, word2vec, tfidf_vectorizer)

    if type == 'mean_emb': return mean_emb
    if type == 'concat_emb': return concat_emb
    if type == 'cv': return cv_emb
    if type == 'tfidf': return tf_emb

# -------------------------------------------------------------------------------------------------------------
def getEmbeddings(data, labels, vocabulary_inv, type):
    """ Clean, randomize, and embed the data """

    data = clean_data(transform_data(vocabulary_inv, data))
    questions, labels = get_random_samples(data, labels)

    # Get embeddings
    embeddings = produceEmbeddings(type, questions)
    return embeddings, labels
