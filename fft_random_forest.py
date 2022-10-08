import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from scipy.fft import rfft, rfftfreq
from sklearn.ensemble import RandomForestClassifier
import pandas as pd


class FFTRandomForest(BaseEstimator, ClassifierMixin):
    def __init__(self, n=20):
        self.n = n

    def fit(self, X, y):
        X, y = check_X_y(X, y)

        self.classes_ = unique_labels(y)

        yf = rfft(X, self.n)
        df = pd.DataFrame(np.abs(yf))

        self.clf = RandomForestClassifier()
        self.clf.fit(df, y)
        return self

    def predict(self, X):
        check_is_fitted(self)
        X = check_array(X)

        yf = rfft(X, self.n)
        df = pd.DataFrame(np.abs(yf))

        return self.clf.predict(df)

    def predict_proba(self, X):
        check_is_fitted(self)
        X = check_array(X)

        yf = rfft(X, self.n)
        df = pd.DataFrame(np.abs(yf))

        return self.clf.predict_proba(df)
