import pandas as pd
import numpy as np
import cPickle
import itertools
from collections import Counter

from utils import load_data

#-------------------------------------------------------------------------------------------------------------
qtype = pd.read_csv("../Data/Question Type.csv")         # 3846 x 3
qtype = qtype.drop_duplicates()                          # 2842 x 3

# Turn the varaibles into categorical and then use one-hot encoding
qtype['qual_cc'] = qtype['Qualitative Tag'].astype('category')
qtype['fund_cc'] = qtype['Fundamental Question'].astype('category')
qtype['qual_codes'] = qtype['qual_cc'].cat.codes
qtype['fund_codes'] = qtype['fund_cc'].cat.codes

# Extract the questions and the labels
questions = qtype.Question.tolist()     # 2842 questions
labels = qtype.qual_codes.tolist()      # 58 labels
questions = np.array(questions)
labels = np.array(labels)

#-------------------------------------------------------------------------------------------------------------
def build_vocab(sentences):
    """
    Takes in a list of lists of tokens.
    """
    # Build vocabulary
    word_counts = Counter(itertools.chain(*sentences))
    # Mapping from index to word
    vocabulary_inv = [x[0] for x in word_counts.most_common()]
    # Mapping from word to index
    vocabulary = {x: i for i, x in enumerate(vocabulary_inv)}
    print len(vocabulary), len(vocabulary_inv)
    return [vocabulary, vocabulary_inv]

def vocab_to_word2vec(filename, vocab, k=300):
    """
    Load word2vec from Mikolov
    """
    word_vecs = {}
    with open(filename, "rb") as f:
        header = f.readline()
        vocab_size, layer1_size = map(int, header.split())
        binary_len = np.dtype('float32').itemsize * layer1_size
        for _ in xrange(vocab_size):
            word = []
            while True:
                ch = f.read(1)
                if ch == ' ':
                    word = ''.join(word)
                    break
                if ch != '\n':
                    word.append(ch)
            word = word.lower()
            if word in vocab:
                word_vecs[word] = np.fromstring(f.read(binary_len), dtype='float32')
            else:
                f.read(binary_len)
    
    print str(len(word_vecs))+" words found in word2vec."
    return word_vecs


def build_word_embedding_mat(word_vecs, vocabulary_inv, k=300):
    """
    Get the word embedding matrix, of size(vocabulary_size, word_vector_size)
    ith row is the embedding of ith word in vocabulary
    """
    vocab_size = len(vocabulary_inv)
    embedding_mat = np.zeros(shape=(vocab_size, k), dtype='float32')
    for idx in xrange(len(vocabulary_inv)):
        #print idx, vocabulary_inv[idx]
        word = vocabulary_inv[idx]
        if word in word_vecs:
            embedding_mat[idx] = word_vecs[word]
        else:
            print word
            embedding_mat[idx] = np.random.uniform(-0.25, 0.25, k)
    print "Embedding matrix of size "+str(np.shape(embedding_mat))
    #initialize the first row,
    #embedding_mat[0]=np.random.uniform(-0.25, 0.25, k)
    return embedding_mat

def build_input_data(sentences, labels, vocabulary):
    """
    Maps sentencs and labels to vectors based on a vocabulary.
    """
    x = [[vocabulary[word] for word in sentence] for sentence in sentences]
    y = np.array(labels)
    return [np.array(x), y]


#-------------------------------------------------------------------------------------------------------------
sentences = load_data(questions)
vocabulary, vocabulary_inv = build_vocab(sentences)
word2vec = vocab_to_word2vec("../Embeddings/GoogleNews-vectors-negative300.bin", vocabulary)
embedding_mat = build_word_embedding_mat(word2vec, vocabulary_inv) #i-th row corresponds to i-th word in vocab
x, y = build_input_data(sentences, labels, vocabulary)

# dump data into pickle format
cPickle.dump([x, y, embedding_mat], open('../Data/train_mat_filtered.pkl', 'wb'))
cPickle.dump(word2vec, open('../Data/word2vec.pkl', 'wb'))
cPickle.dump(vocabulary, open('../Data/vocab_filtered.pkl', 'wb'))
cPickle.dump(vocabulary_inv, open('../Data/vocab_inv_filtered.pkl', 'wb'))



