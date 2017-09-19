import re, string, random
import cPickle
from nltk.tokenize import RegexpTokenizer

# -------------------------------------------------------------------------------------------------------------
tokenizer = RegexpTokenizer(r'\w+')

def clean_str(string, TREC=False):
    """
    Tokenization/string cleaning for the dataset
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)     
    string = re.sub(r"\'s", " \'s", string) 
    string = re.sub(r"\'ve", " \'ve", string) 
    string = re.sub(r"n\'t", " n\'t", string) 
    string = re.sub(r"\'re", " \'re", string) 
    string = re.sub(r"\'d", " \'d", string) 
    string = re.sub(r"\'ll", " \'ll", string) 
    string = re.sub(r",", " , ", string) 
    string = re.sub(r"!", " ! ", string) 
    string = re.sub(r"\(", " \( ", string) 
    string = re.sub(r"\)", " \) ", string) 
    string = re.sub(r"\?", " \? ", string) 
    string = re.sub(r"\s{2,}", " ", string)    
    return string.strip() if TREC else string.strip().lower()

def tokenize(text):
    #split on white spaces, remove punctuation, lower case each word
    return [clean_str(word) for word in tokenizer.tokenize(text)]

def load_data(data):

    txts = []
    for sentence in data:
        txt = tokenize(sentence)
        txts.append(txt)
            
    return txts

# -------------------------------------------------------------------------------------------------------------
def transform_data(vocabulary_inv, data):
    """ Transform data into the actual text of the reviews """
    for i, sample in enumerate(data):
        data[i] = [vocabulary_inv[el] for el in data[i]]
        data[i] = " ".join(data[i])
    return data


def clean_data(data):
    """ Helper function to remove digits, punctuation, and/or stop words """
    for i, sample in enumerate(data):
        data[i] = data[i].translate(None, string.digits)  # remove digits
        data[i] = data[i].translate(None, string.punctuation)
    return data


def get_random_samples(data, labels):
    """ Randomly shuffle questions and labels """
    indices = random.sample(xrange(len(data)), len(data))
    return data[indices], labels[indices]
