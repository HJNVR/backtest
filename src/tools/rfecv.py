import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFECV

class RFECVSelector(object):
    def __init__(self, k) -> None:
        self.k = k
        self.fs = RFECV(LinearRegression(), step=1, cv=5)

    def fit(self, x, y):
        self.fs.fit(x, y)
        self.feature_ranks = self.fs.ranking_
        self.feature_ranks_with_idx = enumerate(self.feature_ranks)
        self.sorted_ranks_with_idx = sorted(self.feature_ranks_with_idx, key=lambda x: x[1])
        self.select_feature_idx = [idx for idx, rnk in self.sorted_ranks_with_idx[:self.k]]