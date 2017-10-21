# Insight

A text classification tool for chatbot utterances. Inputs are questions asked by humans to chatbots, together with the labels other humans provided. The models in this package learn to classify the incoming queries into one of those tags.

<b>Preprocessing</b><br /> Digits and punctuation marks have been removed, words have been tokenized, and sentences have been padded

<b>Embeddings</b><br /> Few different embedding models have been tried. Users can choose between <b>Bag of Words</b>, <b>Tf-idf weightings</b>, as well as more advanced word embeddings such as <b>word2vec</b> or <b>GloVe</b>

<b>Classification</b><br /> We have tried multiple different models. The most successful linear model was Logistic Regression. Users can also choose running a Convolutiona Neural Network (from Yoon Kim 2014 paper) or a Bi-directional LSTM model.

<b>Usage</b><br /> Users can choose whach emebedding models and classification algorithms to un, and whether they want to see a confusion matrix for predicted labels
