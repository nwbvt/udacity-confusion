"""
Helper functions for confusion project
"""
from pandas import DataFrame
from matplotlib import pyplot
import numpy as np
from sklearn.cross_validation import train_test_split, cross_val_score, LabelKFold
from sklearn.metrics import accuracy_score, brier_score_loss, roc_curve
from IPython.display import display, Markdown

class FigureNum(object):
    """
    Quick function for getting the next figure number
    """
    def __init__(self, label="Figure"):
        self._i = 0
        self.label = label
                                            
    def next(self):
        self._i+=1
        return self._i
                                                                    
    def __str__(self):
        return "### " + self.label + " %i" %self.next()
       
    def markdown(self):
        return Markdown(str(self))
                                                                                            
    def __int__(self):
        return self.next()

figure_num = FigureNum()
table_num = FigureNum(label="Table")


def part_f(f, divisions, part):
    """
    Run f against a part of the data
    @param f: the function to run
    @param divisions: the number of divisions
    @param part: the part from 0 to divisions - 1 to run the function against
    @return: the aggregation function
    """
    def agg(data):
        n = len(data)
        start = part * n / divisions
        end = (part + 1) * n / divisions
        return f(data[start:end])
    return agg


def get_probability_scores(classifier, features):
    """
    Get the probability of the positive class
    """
    pos_index = classifier.classes_.tolist().index(1)
    return classifier.predict_proba(features)[:,pos_index]

def brier_score(classifier, features, target):
    """
    Find the brier loss score for the classifier
    """
    probabilities = get_probability_scores(classifier, features)
    return brier_score_loss(target, probabilities)

def test_model(classifier, features, target):
    """
    Run the given model against our data
    Report the mean accuracy (high is good) and the mean brier score (low is good)
    """
    accuracy_scores = cross_val_score(classifier, features, target, cv=40, scoring="accuracy")
    brier_scores = cross_val_score(classifier, features, target, cv=40, scoring=brier_score)
    return np.mean(accuracy_scores), np.mean(brier_scores)

def test_fold(fold, classifier, features, target):
    """Test the accuracy of a model against a given fold"""
    train_indexes, test_indexes = fold
    x_train = features.as_matrix()[train_indexes]
    x_test = features.as_matrix()[test_indexes]
    y_train = target.as_matrix()[train_indexes]
    y_test = target.as_matrix()[test_indexes]
    classifier.fit(x_train, y_train)
    predicted = classifier.predict(x_test)
    return accuracy_score(y_test, predicted)


def test_across_students(classifier, features, target, title_post=""):
    """Tests a classifier across different students"""
    student_labels = map(lambda x: x[0], features.index.tolist())
    student_k_fold = LabelKFold(student_labels, 10)
    student_results = [test_fold(fold, classifier, features, target) for fold in student_k_fold]
    student_results.reverse()
    print "Across different students:"
    print "\tmean: %f, standard deviation: %f" % (np.mean(student_results), np.std(student_results))
    student_plot = pyplot.figure()
    student_plot.suptitle("Figure %i: Accuracy Across Students %s" % (figure_num, title_post))
    pyplot.bar(range(10), student_results)
    pyplot.ylim((0,1.0))


def test_across_videos(classifier, features, target, title_post=""):
    """Tests a classifier aginst different videos"""
    video_labels = map(lambda x: x[1], features.index.tolist())
    video_k_fold = LabelKFold(video_labels, 10)
    video_results = [test_fold(fold, classifier, features, target) for fold in video_k_fold]
    video_results.reverse()
    print "Across different videos:"
    print "\tmean: %f, standard deviation: %f" % (np.mean(video_results), np.std(video_results))
    video_plot = pyplot.figure()
    video_plot.suptitle("Figure %i: Accuarcy Across Videos %s" % (figure_num, title_post))
    pyplot.bar(range(10), video_results)
    pyplot.ylim(0,1.0)
    pyplot.show()


def show_feature_importances(classifier, features, target):
    """
    Show the importance of features for a classifier
    """
    classifier.fit(features, target)
    df = DataFrame(classifier.feature_importances_, index=features.columns).sort_values(0, ascending=False)
    df.columns = ["Importance"]
    display(table_num.markdown(), df)

def draw_roc(classifier, x, y, title):
    """
    Draw the receiver operating curve
    """
    train_x, test_x, train_y, test_y = train_test_split(x, y, random_state=42)
    classifier.fit(train_x, train_y)
    probabilities = get_probability_scores(classifier, test_x)
    fpr, tpr, thresholds = roc_curve(test_y, probabilities)
    roc_plot = pyplot.figure()
    roc_plot.suptitle(title)
    pyplot.plot(fpr, tpr, label="ROC curve")
    pyplot.plot([0,1], [0,1], linestyle="--")
    pyplot.xlabel('False Positive Rate')
    pyplot.ylabel('True Positive Rate')
    pyplot.legend(loc="lower right")
    pyplot.show()

def plot_features(features, target, x_feature, y_feature):
    """
    Plots the features 
    """
    pos_features = features[target == 1.0]
    neg_features = features[target == 0.0]
    pyplot.scatter(pos_features[x_feature], pos_features[y_feature], marker='+', c='r', label='confused')
    pyplot.scatter(neg_features[x_feature], neg_features[y_feature], marker='o', c='b', label='not confused')
    pyplot.xlabel(x_feature)
    pyplot.ylabel(y_feature)
    pyplot.legend(loc='lower right')
    pyplot.suptitle("Figure {figure_num}: Features {x} vs {y}".format(figure_num=figure_num,
                                                                      x=x_feature, y=y_feature))
    pyplot.show()
    
