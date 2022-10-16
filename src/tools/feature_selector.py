from multiprocessing.sharedctypes import Value
from .boruta_lgb import BorutaLGB
from .mutual_info import MISelector
from .FAE import fae_selector
from .rfecv import RFECVSelector
from lightgbm import LGBMRegressor
from sklearn.feature_selection import RFECV
import numpy as np

def feature_selection(x, y, method='boruta', k=25):
    # boruta
    if method == 'boruta':
        lgb = LGBMRegressor(num_boost_round=100)
        feat_selector = BorutaLGB(lgb, n_estimators='auto', verbose=0, random_state=1)
        feat_selector.fit(x, y)
        # Check the selected features
        # selected_feat_cols = [
        #     feat_cols[feat_idx] for feat_idx in range(len(feat_cols)) if feat_selector.support_[feat_idx]]
        selected_x = x[:, feat_selector.support_]
        return selected_x, feat_selector.support_
    # MI-filter
    elif method == 'mi':
        selector = MISelector(k)
        selector.fit(x, y)
        selected_x = x[:, selector.select_feature_idx]
        # selected_feat_cols = [feat_cols[feat_idx] for feat_idx in range(len(feat_cols)) if feat_idx in selector.select_feature_idx]
        return selected_x, selector.select_feature_idx
    elif method == 'ae':
        selector = fae_selector(x, k)
        tmp = selector.predict(x)
        selected_ids = np.where(np.sum(tmp, axis=0) != 0)[0]
        selected_x = x[:, selected_ids]
        return selected_x, selected_ids
    # RFECV: Recursive feature elimination with cross-validation
    elif method == 'rfecv':
        selector = RFECVSelector(k)
        selector.fit(x,y)
        selected_x = x[:, selector.select_feature_idx]
        return selected_x, selector.select_feature_idx
    elif method == '':
        return x, np.arange(x.shape[1])
    else:
        raise ValueError('Invalid feature selection method...')
