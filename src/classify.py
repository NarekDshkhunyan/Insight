#from sklearn.model_selection import GridSearchCV
import cPickle

from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn import svm, tree
from sklearn.neighbors import NearestNeighbors, RadiusNeighborsClassifier, KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
METRICS_CHOICE = 'weighted'

#-------------------------------------------------------------------------------------------------------------
def classify(method, X_train, y_train, X_test, y_test, features, results):
    """ Runs the classification algorithm of the choice """

    #start = time.time()
    if method == "mnb":
        clf = MultinomialNB(alpha=0.5, class_prior=[0.1, 0.1, 0.1, 0.25, 0.45])
    elif method == "svm":
        # param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5], 'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1]}
        # clf = GridSearchCV(svm.SVC(kernel='rbf', class_weight='balanced'), param_grid)
        clf = svm.SVC(C=1000.0, kernel="rbf")
    elif method == "knn":
        clf = KNeighborsClassifier(n_neighbors=10, algorithm='auto')
    elif method == "dt":
        clf = tree.DecisionTreeClassifier(min_samples_split=30)
    elif method == "rf":
        clf = RandomForestClassifier(n_estimators=100, max_features='sqrt', oob_score=True, n_jobs=-1, random_state=42)
    elif method == "lr":
        clf = LogisticRegression(C=0.1, class_weight='balanced', solver='newton-cg', multi_class='multinomial', n_jobs=-1,
                                random_state=42)

    # Fit the model and predict labels
    clf = clf.fit(X_train, y_train)
    y_predicted = clf.predict(X_test)

    results = evaluate(y_test, y_predicted, results)



#-------------------------------------------------------------------------------------------------------------
def evaluate(y_test, y_predicted, results):
    """ Helper function to print the results """
    precision = precision_score(y_test, y_predicted, pos_label=None,
                                average=METRICS_CHOICE)  # true positives / (true positives+false positives)
    recall = recall_score(y_test, y_predicted, pos_label=None,
                          average=METRICS_CHOICE)  # true positives /(true positives + false negatives)
    f1 = f1_score(y_test, y_predicted, pos_label=None, average=METRICS_CHOICE)
    accuracy = accuracy_score(y_test, y_predicted)  # num of correct predictions/ total num of predictions
    print "accuracy = %.3f, precision = %.3f, recall = %.3f, f1 = %.3f" % (accuracy, precision, recall, f1)
    results['accuracy'].append(accuracy)
    results['precision'].append(precision)
    results['recall'].append(recall)
    results['f1'].append(f1)
    return results


results = {'accuracy': [], 'precision': [], 'recall': [], 'f1': []}
results_random = {'accuracy': [], 'precision': [], 'recall': [], 'f1': []}

input_file = "../Data/input_data.pkl"
with open(input_file) as f:
    X_train, X_test, y_train, y_test, features = cPickle.load(f)

# Run the classification algorithm
classify('rf', X_train, y_train, X_test, y_test, features, results)
