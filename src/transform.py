import string
import random
import cPickle
from embeddings import produceEmbeddings
from sklearn.model_selection import train_test_split

#-------------------------------------------------------------------------------------------------------------
def transform_data(data):
    """ Tranform data into the actual text of the reviews """
    for i, sample in enumerate(data):
        data[i] = [vocabulary_inv[el] for el in data[i]]
        data[i] = " ".join(data[i])
    return data

def clean_data(data):
    """ Helper function to remove digits, punctuation, and/or stop words """
    for i, sample in enumerate(data):
        data[i] = data[i].translate(None, string.digits)                          # remove digits
        data[i] = data[i].translate(None, string.punctuation)
    return data

def get_random_samples(data, labels):
    """ Randomly shuffle questions and labels """
    indices = random.sample(xrange(len(data)), len(data))
    return data[indices], labels[indices]


#-------------------------------------------------------------------------------------------------------------
data_file = "../Data/train_mat_filtered.pkl"
vocab_file = "../Data/vocab_filtered.pkl"
vocab_inv_file = "../Data/vocab_inv_filtered.pkl"

with open(data_file) as f:
    data, labels, _ = cPickle.load(f)

with open(vocab_file) as f:
    vocabulary = cPickle.load(f)

with open(vocab_inv_file) as f:
    vocabulary_inv = cPickle.load(f)

data = clean_data(transform_data(data))
questions, labels = get_random_samples(data, labels)
# Random split - think about KFold as well
X_train, X_test, y_train, y_test = train_test_split(questions, labels, test_size=0.1, random_state=42)

# Get embeddings
X_train, X_test, features = produceEmbeddings(X_train, X_test)
cPickle.dump([X_train, X_test, y_train, y_test, features], open('../Data/input_data.pkl', 'wb'))
