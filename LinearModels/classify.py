from sklearn import svm, tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
METRICS_CHOICE = 'weighted'

#-------------------------------------------------------------------------------------------------------------
def classify(method, X_train, y_train, X_test, y_test, results):
    """ Runs the classification algorithm of the choice """

    if method == "svm":
        param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5], 'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], 'kernel': ('linear', 'rbf')}
        #clf = GridSearchCV(svm.SVC(class_weight='balanced'), param_grid)
        clf = svm.SVC(C=5000.0, gamma=0.0005, kernel='rbf')
    elif method == "lr":
        param_grid = {'C': [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0]}
        #clf = GridSearchCV(LogisticRegression(class_weight='balanced', solver='newton-cg', multi_class='multinomial', n_jobs=-1, random_state = 42), param_grid)
        clf = LogisticRegression(C=30.0, class_weight='balanced', solver='newton-cg', multi_class='multinomial', n_jobs=-1, random_state=42)

    elif method == "knn":
        clf = KNeighborsClassifier(n_neighbors=10, algorithm='auto')
    elif method == "dt":
        clf = tree.DecisionTreeClassifier(min_samples_split=30)
    elif method == "rf":
        clf = RandomForestClassifier(n_estimators=100, max_features='sqrt', oob_score=True, n_jobs=-1, random_state=42)

    # Fit the model and predict labels
    clf = clf.fit(X_train, y_train)
    y_predicted = clf.predict(X_test)

    evaluate(y_test, y_predicted, results)

    return y_predicted



#-------------------------------------------------------------------------------------------------------------
def evaluate(y_test, y_predicted, results):
    """ Helper function to print the results """

    precision = precision_score(y_test, y_predicted, pos_label=None,
                                average=METRICS_CHOICE)             # true positives / (true positives+false positives)
    recall = recall_score(y_test, y_predicted, pos_label=None,
                          average=METRICS_CHOICE)                   # true positives / (true positives + false negatives)
    f1 = f1_score(y_test, y_predicted, pos_label=None, average=METRICS_CHOICE)
    accuracy = accuracy_score(y_test, y_predicted)                  # num of correct predictions/ total num of predictions
    print "accuracy = %.3f, precision = %.3f, recall = %.3f, f1 = %.3f" % (accuracy, precision, recall, f1)
    results['accuracy'].append(accuracy)
    results['precision'].append(precision)
    results['recall'].append(recall)
    results['f1'].append(f1)
    return results
