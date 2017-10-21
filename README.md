# Insight

A text classification tool for chatbot utterances. Inputs are questions asked by humans to chatbots, together with the labels other humans provided. The models in this package learn to classify the incoming queries into one of those tags.

<b>Preprocessing</b><br /> Digits and punctuation marks have been removed, words have been tokenized, and sentences have been padded

<b>Embeddings</b><br /> Few different embedding schemes have been tried. Users can choose between <b>Bag of Words</b>, <b>Tf-idf weightings</b>, as well as more advanced word embeddings such as <b>word2vec</b> or <b>GloVe</b>

<b>Classification</b><br /> We have tried multiple different models. The most successful linear model was Logistic Regression. Users can also try neural nets, such as Convolutiona Neural Network (from Yoon Kim 2014 paper) or a Bi-Directional LSTM model.

<b>Usage</b><br /> Users can choose which emebedding model and classification algorithm to run, and whether they want to see a confusion matrix for predicted labels

To run a linear model, do "cd ../LinearModels", then "python main.py lr tfidf False" for the best model at the moment. You can also customize 

+ *algorithm* 'lr' for Logistic Regression, 'svm' for Support Vector Machines, 'knn' for k-Nearest Neighbors, and 'rf' for Random Forest

+ *embedding* 'cv' for Bag-of-Words, 'tfidf' for Tf-idf scores, 'mean-emb' for averaged word2vec

+ *plot_confusion_matrix* 'True' to plot a confusion matrix for predicted labels

To run a deep neural network model, do "cd ../DeepModels", then "python main.py attention False". You can also customize

+ *algorithm"* 'ConvNet' for Yoon Kim 2014 CNN architecture, 'LSTM' for a bi-directional lstm, 'Attention' for bi-directional lstm with attention

+ *plot_confusion_matrix"* 'True' to plot a confusion matrix for predicted labels
