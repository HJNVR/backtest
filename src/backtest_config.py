# read from ./config/backtest.json file 
# set corresponding result_kpi path for specific paper_id
# set corresponding summary.csv path for specific paper_id 

# ============================================================================

import os
import json
import optuna
import numpy as np
import pandas as pd

class BacktestConfig:
    def __init__(self, lr_sample=False, lgb_sample=False):
        self.is_lr_sample = lr_sample
        self.is_lgb_sample = lgb_sample
        # check back_test.json in config folder for more details
        self.args = json.load(open('../config/back_test.json', 'r')) 
        if self.is_lr_sample:
            self.args = json.load(open('../config/back_test_sample_lr.json', 'r'))  
        if self.is_lgb_sample:
            self.args = json.load(open('../config/back_test_sample_lgb.json', 'r'))  

        # Get all file paths configs
        self.result_path = self.args['file_paths']['result_path']
        self.sp500_historical_list_path = self.args['file_paths']['sp500_historical_list_path']
        self.spy_daily_prc_path = self.args['file_paths']['spy_daily_prc_path']
        self.sp500_daily_prc_path = self.args['file_paths']['sp500_daily_prc_path']
        self.paper1_features_path = self.args['file_paths']['paper1_features_path']
        self.paper3_features_path = self.args['file_paths']['paper3_features_path']
        self.paper4_features_path = self.args['file_paths']['paper4_features_path']
        self.paper6_features_path = self.args['file_paths']['paper6_features_path']
        self.paper7_features_path = self.args['file_paths']['paper7_features_path']
        self.paper9_features_path = self.args['file_paths']['paper9_features_path']
        self.paper11_features_path = self.args['file_paths']['paper11_features_path']
        self.consolidated_features_path = self.args['file_paths']['consolidated_features_path']
        self.paper1_final_summary_path = self.args['file_paths']['paper1_final_summary_path']
        self.paper3_final_summary_path = self.args['file_paths']['paper3_final_summary_path']
        self.paper4_final_summary_path = self.args['file_paths']['paper4_final_summary_path']
        self.paper6_final_summary_path = self.args['file_paths']['paper6_final_summary_path']
        self.paper7_final_summary_path = self.args['file_paths']['paper7_final_summary_path']
        self.paper9_final_summary_path = self.args['file_paths']['paper9_final_summary_path']
        self.paper11_final_summary_path = self.args['file_paths']['paper11_final_summary_path']
        self.consolidated_final_summary_path = self.args['file_paths']['consolidated_final_summary_path']
        self.lr_sample_final_summary_path = self.args['file_paths']['lr_sample_final_summary_path']
        self.lgb_sample_final_summary_path = self.args['file_paths']['lgb_sample_final_summary_path']
        self.paper1_kpi_result_path = self.args['file_paths']['paper1_kpi_result_path']
        self.paper3_kpi_result_path = self.args['file_paths']['paper3_kpi_result_path']
        self.paper4_kpi_result_path = self.args['file_paths']['paper4_kpi_result_path']
        self.paper6_kpi_result_path = self.args['file_paths']['paper6_kpi_result_path']
        self.paper7_kpi_result_path = self.args['file_paths']['paper7_kpi_result_path']
        self.paper9_kpi_result_path = self.args['file_paths']['paper9_kpi_result_path']
        self.paper11_kpi_result_path = self.args['file_paths']['paper11_kpi_result_path']
        self.consolidated_kpi_result_path = self.args['file_paths']['consolidated_kpi_result_path']
        self.lr_sample_kpi_result_path = self.args['file_paths']['lr_sample_kpi_result_path']
        self.lgb_sample_kpi_result_path = self.args['file_paths']['lgb_sample_kpi_result_path']
        self.best_combo_chosen_params_path = self.args['file_paths']['best_combo_chosen_params_path']
        self.cv_param_details_path = self.args['file_paths']['cv_param_details_path']
        if not os.path.exists(self.result_path):
            os.makedirs(self.result_path)

        # common_setting configs
        self.paper_id = self.args['common_settings']['paper_id']
        self.debug_mode = self.args['common_settings']['debug_mode']

        # ML parameters configs
        self.y_col_prefix = self.args['ml_parameters']['y_col_prefix']
        self.pred_col = 'pred_ret'
        self.dummy_cols = self.args['ml_parameters']['primary_key']
        self.kfold = self.args['ml_parameters']['kfold']
        self.train_end_ym = self.args['ml_parameters']['first_train_end_ym']
        self.tune_sampler = self.args['ml_parameters']['tune_sampler']
        self.end_date = self.args['ml_parameters']['end_date']
        self.md = self.args['ml_parameters']['md']
        self.fs_tool = self.args['ml_parameters']['fs']
        self.test_period = self.args['ml_parameters']['test_period']
        self.train_period = self.args['ml_parameters']['train_period']
        self.train_valid_period = self.args['ml_parameters']['train_valid_period']
        self.money = self.args['ml_parameters']['money']
        self.topk = self.args['ml_parameters']['topk']
        self.tune_trails = self.args['ml_parameters']['tune_trials']

        if self.tune_sampler == 'tpe':
            self.sampler = optuna.samplers.TPESampler(seed=2020)
        elif self.tune_sampler == 'random':
            self.sampler = optuna.samplers.RandomSampler(seed=2020)
        elif self.tune_sampler == 'nsga':
            self.sampler = optuna.samplers.NSGAIISampler(seed=2020)
        elif tune_sampler == 'cma':
            self.sampler = optuna.samplers.CmaEsSampler(seed=2020)
        else:
            raise ValueError('invalid sampler value')

        # logistic regression parameters
        #self.percentile = self.args['logistic_parameters']['percentile']
        #self.threshold = self.args['logistic_parameters']['threshold']

    def get_kpi_result_path(self):
        if self.is_lr_sample:
            return self.lr_sample_kpi_result_path
        if self.is_lgb_sample:
            return self.lgb_sample_kpi_result_path
        elif self.paper_id.lower() == 'all':
            return self.consolidated_kpi_result_path
        elif self.paper_id.lower() == 'paper1':
            return self.paper1_kpi_result_path
        elif self.paper_id.lower() == 'paper3':
            return self.paper3_kpi_result_path
        elif self.paper_id.lower() == 'paper4':
            return self.paper4_kpi_result_path
        elif self.paper_id.lower() == 'paper6':
            return self.paper6_kpi_result_path
        elif self.paper_id.lower() == 'paper7':
            return self.paper7_kpi_result_path
        elif self.paper_id.lower() == 'paper9':
            return self.paper9_kpi_result_path
        elif self.paper_id.lower() == 'paper11':
            return self.paper11_kpi_result_path
        else:
            raise ValueError('invalid paper id')

    def get_final_summary_path(self):
        if self.is_lr_sample:
            return self.lr_sample_final_summary_path
        elif self.is_lgb_sample:
            return self.lgb_sample_final_summary_path
        elif self.paper_id.lower() == 'all':
            return self.consolidated_final_summary_path
        elif self.paper_id.lower() == 'paper1':
            return self.paper1_final_summary_path
        elif self.paper_id.lower() == 'paper3':
            return self.paper3_final_summary_path
        elif self.paper_id.lower() == 'paper4':
            return self.paper4_final_summary_path
        elif self.paper_id.lower() == 'paper6':
            return self.paper6_final_summary_path
        elif self.paper_id.lower() == 'paper7':
            return self.paper7_final_summary_path
        elif self.paper_id.lower() == 'paper9':
            return self.paper9_final_summary_path
        elif self.paper_id.lower() == 'paper11':
            return self.paper11_final_summary_path
        else:
            raise ValueError('invalid paper id')

    def get_lr_percentile_threshold(self):
        #`logisitc_parameters` : 
        #`percentile` : rank and obtain stocks with the top 20% future return
        # `threshold` : convert future return to 1, if future return >= threshold. Otherwise, convert future return to -1, if future return < threshold.
        if self.is_lr_sample:
            return 0.25, 0.15
        elif self.is_lgb_sample:
            return 0.25, 0.15
        elif self.paper_id.lower() == 'all':
            return 0.1, 0.2
        elif self.paper_id.lower() == 'paper1':
            return 0.2, 0.3
        elif self.paper_id.lower() == 'paper3':
            return 0.25, 0.15
        elif self.paper_id.lower() == 'paper4':
            return 0.05, 0.35
        elif self.paper_id.lower() == 'paper6':
            return 0.05, 0.35
        elif self.paper_id.lower() == 'paper7':
            return 0.2, 0.3
        elif self.paper_id.lower() == 'paper9':
            return 0.25, 0.15
        elif self.paper_id.lower() == 'paper11':
            return 0.25, 0.2
        else:
            raise ValueError('invalid paper id')

if __name__ == "__main__":
    backtestconfig = BacktestConfig()
    print(backtestconfig.args)
