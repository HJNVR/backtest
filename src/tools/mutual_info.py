import numpy as np
from sklearn.feature_selection import mutual_info_regression, SelectKBest

class MISelector(object):
    def __init__(self, k) -> None:
        self.k = k
        self.fs = SelectKBest(score_func=mutual_info_regression, k=k)

    def fit(self, x, y):
        self.fs.fit(x, y)
        self.feature_scores = self.fs.scores_
        self.num_features = self.fs.n_features_in_
        self.select_feature_idx = np.argsort(self.feature_scores)[::-1][:self.k]

    def transform(self, x):
        return self.fs.transform(x)