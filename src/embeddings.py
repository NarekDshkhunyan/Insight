from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

train_set = ("The sky is blue.", "The sun is bright.")
test_set = ("The sun in the sky is bright.", "We can see the shining sun, the bright sun.")

#count_vectorizer = CountVectorizer()
#count_vectorizer.fit_transform(train_set)
#print "Vocabulary:", count_vectorizer.vocabulary_

#freq_term_matrix = count_vectorizer.transform(test_set)
#print freq_term_matrix.todense()

#tfidf = TfidfTransformer(norm='l2')
#tfidf.fit(freq_term_matrix)
#print "IDF:", tfidf.idf_

#tf_idf_matrix = tfidf.transform(freq_term_matrix)
#print tf_idf_matrix.todense()


#-------------------------------------------------------------------------------------------------------------
def produceEmbeddings(train_data, test_data):
    
    count_vectorizer = CountVectorizer()
    train_embeddings = count_vectorizer.fit_transform(train_data)
    test_embeddings = count_vectorizer.transform(test_data)
    features = count_vectorizer.get_feature_names()

    #tfidf = TfidfTransformer(norm='l2')
    #tfidf.fit(freq_term_matrix)
    #tf_idf_matrix = tfidf.transform(freq_term_matrix)
    
    return train_embeddings, test_embeddings, features
