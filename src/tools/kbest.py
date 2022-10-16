import numpy as np
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectKBest
from sklearn.linear_model import LogisticRegression

class RFECVSelector(object):
    def __init__(self, k) -> None:
        self.k = k
        self.fs = SelectKBest(score_func=chi2, k=k)

    def fit(self, x, y):
        self.fs.fit(x, y)
        self.feature_ranks = self.fs.scores_
        self.feature_ranks_with_idx = enumerate(self.feature_ranks)
        self.sorted_ranks_with_idx = sorted(self.feature_ranks_with_idx, key=lambda x: x[1])
        self.select_feature_idx = [idx for idx, rnk in self.sorted_ranks_with_idx[:self.k]]