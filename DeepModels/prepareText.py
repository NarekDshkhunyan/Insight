import cPickle
import numpy as np
import pandas as pd

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical

MAX_SEQUENCE_LENGTH = 19            # 1000
MAX_NB_WORDS = 2000                 # 20000
VALIDATION_SPLIT = 0.1

# prepare text samples and their labels

print('Processing text dataset')

qtype = pd.read_csv("../Data/QuestionType.csv")         # 3846 x 3
qtype = qtype.drop_duplicates()                         # 2842 x 3, 58 labels

# Remove categories with < 20 questions
categories = ['Roxy-Composition-Internet','Hotel-Location','Roxy-About-Sports','Hotel-Permission','Roxy-About-Social_Media','Roxy-Surveillance-Eavesdrop','Roxy-About-Pets',
            'Roxy-History-Persons','Roxy-History-Movement','Roxy-Composition-Water','Conversation-Secrets','Roxy-About-Politics','Conversation-Statement',
            'Roxy-Surveillance-Recording','Hotel-Controls','Roxy-Composition-Internal','Roxy-About-Gender','Roxy-History-Work','Roxy-History-Memory','Roxy-About-Religion',
            'Roxy-History-Education','Roxy-Power-Charging','Roxy-Surveillance-Camera','Hotel-Question','Roxy-Composition-Connection',
            #'Roxy-Composition-Language','Roxy-Composition-Durability','Hotel-Knowledge','Roxy-Power-Sleep','Roxy-History-Being_in_Room','Roxy-Power-General','Roxy-About-Naming',
            #'Conversation-Other_Persons','Roxy-Power-Battery','Roxy-Function-Boss','Roxy-Surveillance-Listening','Roxy-Composition-Cost','Roxy-Surveillance-Spying'
            ]
for category in categories:
    qtype = qtype[qtype['Qualitative Tag']!=category]

# Turn the varaibles into categorical and then use one-hot encoding
qtype['qual_cc'] = qtype['Qualitative Tag'].astype('category')
qtype['fund_cc'] = qtype['Fundamental Question'].astype('category')
qtype['qual_codes'] = qtype['qual_cc'].cat.codes
qtype['fund_codes'] = qtype['fund_cc'].cat.codes

# Extract the questions and the labels
questions = qtype.Question.tolist()     # 2568 questions
labels = qtype.qual_codes.tolist()      # 33 labels
print len(questions)

print('Found %s texts.' % len(questions))

# finally, vectorize the text samples into a 2D integer tensor
tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(questions)
sequences = tokenizer.texts_to_sequences(questions)

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

labels = to_categorical(np.asarray(labels))
print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', labels.shape)

# split the data into a training set and a validation set
indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]
num_validation_samples = int(VALIDATION_SPLIT * data.shape[0])

x_train = data[:-num_validation_samples]
y_train = labels[:-num_validation_samples]
x_val = data[-num_validation_samples:]
y_val = labels[-num_validation_samples:]

# dump data into pickle format
cPickle.dump([x_train, y_train], open('../Data/train_matrix.pkl', 'wb'))
cPickle.dump([x_val, y_val], open('../Data/val_matrix.pkl', 'wb'))
cPickle.dump(word_index, open('../Data/word_index.pkl', 'wb'))
