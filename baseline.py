from sklearn.utils import shuffle
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.metrics import classification_report, recall_score, make_scorer, f1_score
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.utils import shuffle

def parse_line(line):
    line = line.split("\t")
    return line[0], line[1], int(line[2] == "OFF")

def evaluate_baseline(_set):

    print("Evaluating", _set)
    with open("OffensEval/offenseval-" + _set + "-training-v1.tsv") as fi:
        lines = fi.read().splitlines()
        ids, x_train, y_train = zip(*list(map(lambda x: parse_line(x), lines[1:]))) 

    X, x_dev, y, y_dev = train_test_split(x_train, y_train, random_state=1234, test_size=0.25)

    bow = CountVectorizer(max_features=2000)
    tfidf = TfidfTransformer()

    svm_clf = SVC(C=10, gamma='scale', kernel='linear')
    cnb_clf = ComplementNB()

    pipeline = Pipeline([('bow', bow),
                        ('tfidf', tfidf),
                        ('clf', svm_clf),])

    print('training on', len(X), 'samples')
    pipeline.fit(X, y)

    predictions = pipeline.predict(X)
    print ('Training data\n', classification_report(y, predictions))

    predictions = pipeline.predict(x_dev)
    print ('Dev data\n', classification_report(y_dev, predictions))


def main():
    for _set in ("ar", "da", "greek", "tr"):
        evaluate_baseline(_set)

if __name__ == "__main__":
    main()