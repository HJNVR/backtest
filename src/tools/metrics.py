import numpy as np
from sklearn.metrics import accuracy_score


# 1 - wmape is accuracy
def wmape(ground_truth, predict): # small is better
    ground_truth = np.array(ground_truth)
    predict = np.array(predict)
    return np.abs(predict - ground_truth).sum() / np.abs(ground_truth).sum()


def oos_rsquare(ground_truth, predict): # lager is better
    return 1 - (np.power(ground_truth - predict, 2).sum()/ np.power(ground_truth, 2).sum())

def accuracy(y_pred, y_truth):
    score =accuracy_score(y_pred, y_truth)
    return score