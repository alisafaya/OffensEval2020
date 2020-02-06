from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np

class RecallBiasedEstimator(BaseEstimator, ClassifierMixin):

    def __init__(self, classifiers):
        self.classifiers = classifiers

    def fit(self, X, y):
        # fit each model individually
        for clf in self.classifiers:
            clf.fit(X, y)

        return self

    def predict(self, X):

        # Get prediction from each model
        preds = [ clf.predict(X) for clf in self.classifiers ]

        # If any individual classifier predicts positive results then the final result is positive
        results = np.zeros((X.shape[0],))
        for x in range(X.shape[0]):
            pred = sum([ ind_clf_result[x] for ind_clf_result in preds ])
            if pred > 0:
                results[x] = 1

        return results
