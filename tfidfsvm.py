from sklearn.metrics import classification_report, recall_score, make_scorer, f1_score
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.utils import shuffle
from sklearn.svm import SVC
from data import *

def evaluate_baseline(_set):
    print("Building baseline for:", _set)

    train_samples = read_file(_set +".train")
    X, y = [ x["text"] for x in train_samples ], [ x["label"] for x in train_samples ]
    
    bow = CountVectorizer(max_features=3000)
    tfidf = TfidfTransformer()

    svm_clf = SVC(C=10, gamma='scale', kernel='linear')

    pipeline = Pipeline([('bow', bow),
                        ('tfidf', tfidf),
                        ('clf', svm_clf),])

    print('\tTraining on', len(X), 'samples')
    pipeline.fit(X, y)

    predictions = pipeline.predict(X)
    print ('-'* 40, '\nTraining data\n', classification_report(y, predictions, digits=3))

    # Testing
    print("Evaluating SVM classifier")
    test_samples = read_file(_set +".test")
    X, y = [ x["text"] for x in test_samples ], [ x["label"] for x in test_samples ]

    predictions = pipeline.predict(X)
    print ('Test data\n', classification_report(y, predictions, digits=3))

def main():
    for _set in ("ar", "gr", "tr"):
        evaluate_baseline(_set)

if __name__ == "__main__":
    main()