# main function
# train -> backtest process
# plot -> plot backtest results

# ============================================================================
import os
import time
import json
import optuna
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from itertools import product
from dateutil.relativedelta import relativedelta
from lightgbm import LGBMRegressor
from sklearn.linear_model import LogisticRegression
import warnings; warnings.filterwarnings('ignore')

from backtest_config import BacktestConfig
from dataset import BacktestDataset

from tools.metrics import oos_rsquare
from tools.balance_data import downsampling
from tools.model_selection import rolling_valid_split
from tools.feature_selector import feature_selection

class BackTest:
    def __init__(self):
        self.backtest_config = BacktestConfig(lr_sample=True)
        self.backtest_dataset = BacktestDataset(lr_sample=True)

        # legal ticker at certain month should be sold for a specific period (fixed length)
        self.daily_prc = self.backtest_dataset.get_daily_prc()
        self.counting_max =  self.daily_prc.groupby(['ticker', 'year', 'month'])['date'].count().reset_index()
        # counting_max will be used to validate ticker when selecting top-k tickers
        self.counting_max = self.counting_max.groupby(['year', 'month'])['date'].max()
        # sp500 historical list
        self.sp500_historical_list = self.backtest_dataset.get_sp500_historical_list()

        # target paper features
        self.df = self.backtest_dataset.get_paper_features()
        self.daily_prc = self.backtest_dataset.get_daily_prc()
        self.pf_daily_trend = pd.DataFrame()
        #self.df_copy = self.df.copy()

         # get necessary config parameters
        self.train_end_date = pd.to_datetime(self.backtest_config.train_end_ym, format='%Y%m')
        self.train_valid_period = self.backtest_config.train_valid_period
        self.test_period = self.backtest_config.test_period
        self.train_period = self.backtest_config.train_period
        self.fs_tool = self.backtest_config.fs_tool
        self.topk = self.backtest_config.topk
        self.kfold = self.backtest_config.kfold
        self.y_col_prefix = self.backtest_config.y_col_prefix
        self.money = self.backtest_config.money
        self.md = self.backtest_config.md
        self.sampler = self.backtest_config.sampler

        self.y_cols = [f'fut_ret{col}' for col in self.test_period]
        self.dummy_cols = self.backtest_config.dummy_cols + ['trade_date']
        self.feat_cols = [col for col in self.df.columns if col not in (self.dummy_cols + self.y_cols)]

        # debug 
        if self.backtest_config.debug_mode.lower() == 't':
            self.debug_mode = True
            self.best_combo_chosen_params = pd.DataFrame(columns=['best_combo', 'train_start_date', 'actual_train_end_date', 'pred_date',
                                                 'max_mean_oos_square'])
            self.cv_param_details = pd.DataFrame(columns=['current_combo', 'valid_train_start_date',' valid_train_end_date', 'valid_end_date', 'oos_square'])
        else:
            self.debug_mode = False


    # percentile, threshold data processing 
    def choose_by_percentile(self, df, percentile, threshold): 
        for year in tqdm(range(pd.DatetimeIndex(df['date'].head(1)).year[0], pd.DatetimeIndex(df['date'].tail(1)).year[0]+1)):
            for month in range(1,13):
                current_df = df[(pd.DatetimeIndex(df['date']).year == year) & (pd.DatetimeIndex(df['date']).month == month)]
                current_df = current_df.sort_values(by=['fut_ret1'], ascending=False)
                def convert_to_categorical():
                    for i in current_df.index:
                        if i in current_df.head(int(percentile*current_df.shape[0])).index and \
                        current_df.loc[i]['fut_ret1'] >= threshold:
                            df.at[i, 'fut_ret1'] = 1
                        else:
                            df.at[i, 'fut_ret1'] = -1
                convert_to_categorical()

    # cv hyperparameters for different models
    def get_cv_hyperparameters(self, md, trial):
        # logisitc_regression hyperparameters | could add more with more computation power
        if md.lower() == 'lr':
            param = {
            'penalty' :  trial.suggest_categorical('penalty', ['l1', 'l2']),
            'tol' : trial.suggest_uniform('tol' , 1e-6 , 1e-3),
            'C' : trial.suggest_loguniform("C", 1e-2, 1),
            'solver' : trial.suggest_categorical('solver', ['liblinear', 'saga']),
            'max_iter' : trial.suggest_int('max_iter', 100, 500)
            }

        # lightgbm hyperparameters
        elif md.lower() == 'lgb':
            param = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 500),   
                'num_leaves': trial.suggest_int('num_leaves', 10, 512),
                'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 10, 80),
                'bagging_fraction': trial.suggest_float('bagging_fraction', 0.0, 1.0), # subsample
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),  # eta
                'lambda_l1': trial.suggest_float('lambda_l1', 0.01, 1),  # reg_alpha
                'lambda_l2': trial.suggest_float('lambda_l2', 0.01, 1), # reg_lambda
            }
        return param

    # main train 
    def train(self):
        if self.md.lower() == 'lr': # sort and get top percentile stocks
            self.choose_by_percentile(self.df, self.backtest_config.percentile, self.backtest_config.threshold)
            
        pred_col = 'pred_ret'

        # record all the important results in a csv file
        #f = open(f'self.backtest_config.final_summary_path', 'w', encoding='utf-8')
        f = open(self.backtest_config.get_final_summary_path(), 'w', encoding='utf-8')
        f.write('start date,end date,train_valid_period(years),test_period(months),train_period(years),topk,feature_selection,Annual Return,MDD,Calmar Ratio\n')

        while self.train_end_date < pd.to_datetime(self.backtest_config.end_date, format='%Y%m'):
            if self.debug_mode: best_combo_chosen_list = []
            print(f'[combo selection]End date of training is: {self.train_end_date}')
            combo_cnt = 0
            # choose combination of train_valid_date, test_period:
            calmar_ratio_res = pd.DataFrame({
                'train_valid_period':[], 'test_period': [],
                'train_period': [], 'topk': [], 'fs': [], 'cr': []
            })
            ar_res = pd.DataFrame({
                'train_valid_period':[],'test_period': [],
                'train_period': [], 'topk': [], 'fs': [], 'ar': []
            })
            mdd_res = pd.DataFrame({
                'train_valid_period':[], 'test_period': [],
                'train_period': [], 'topk': [], 'fs': [], 'mdd': []
            })

            pf_daily_res = {}
            fund_res = {}
            init_fund = self.money
            combo_kpi = {} # dic of backtest combination and mean oos_square kpi
            def objective(trial):
                param = self.get_cv_hyperparameters(self.md, trial)
                kpis = [] # the kpis records of each backtest combo
                for combo in product(self.train_valid_period, self.test_period, self.train_period, self.topk, self.fs_tool):
                    print('current combination:', combo)
                    train_valid_period = combo[0]
                    test_period = combo[1]
                    train_period = combo[2]
                    topk = combo[3]
                    fs_tool = combo[4]
                    valid_period = test_period
                    # change year to month
                    train_valid_period *= 12
                    train_period *= 12

                    y_col = f'{self.y_col_prefix}{test_period}'
                    data = self.df[self.dummy_cols + self.feat_cols + [y_col]].copy()
                    data = data.dropna()
                    actual_train_end_date = self.train_end_date - relativedelta(months=test_period - 1)
                    train_start_date = self.train_end_date - relativedelta(months=train_valid_period)

                    #param check
                    if train_start_date < pd.to_datetime('200001', format='%Y%m'):
                        raise ValueError('invalid first_train_end_ym value')
                    predict_end_date = self.train_end_date + relativedelta(months=test_period)
                    predict_dates = pd.date_range(
                        start=self.train_end_date + relativedelta(months=1), end=min(predict_end_date, self.df['date'].max()), freq='MS')

                    valid_date_sets = rolling_valid_split(self.kfold, train_period, train_start_date, actual_train_end_date, valid_period)
                    train_data = data.query(f'"{train_start_date}" < date <= "{actual_train_end_date}"')

                    if self.md.lower() == 'lr':
                        # balance data - downsampling
                        train_data = downsampling(train_data)

                    X_train = train_data[self.feat_cols].values
                    y_train = train_data[y_col].values

                    # implement feature selection, 'mi', 'boruta', 'ae', 'rfecv', ''

                    try:
                        if self.backtest_config.paper_id.lower() == 'all':
                            _, select_ids = feature_selection(X_train, y_train, method=fs_tool, k=200)             
                        else:
                            _, select_ids = feature_selection(X_train, y_train, method=fs_tool, k=25)         
                    except:
                        _, select_ids = feature_selection(X_train, y_train, method=fs_tool, k=5)       

                    if sum(select_ids) == 0:
                        select_ids = [True for _ in self.feat_cols]
                        
                    for valid_train_start_date, valid_train_end_date, valid_end_date in valid_date_sets: 
                        fold_train_data = data.query(f'"{valid_train_start_date}" < date <= "{valid_train_end_date}"')
                        if self.md.lower() == 'lr':
                            # balance data - downsampling
                            fold_train_data = downsampling(fold_train_data)

                        fold_valid_data = data.query(f'"{valid_train_end_date}" < date <= "{valid_end_date}"')
                        train_x = fold_train_data[self.feat_cols].values[:, select_ids]
                        train_y = fold_train_data[y_col].values
                        val_x = fold_valid_data[self.feat_cols].values[:, select_ids]
                        val_y = fold_valid_data[y_col].values
                        if self.backtest_config.md == 'lgb':
                            model = LGBMRegressor(seed=2022, **param)
                        elif self.backtest_config.md == 'lr':
                            model = LogisticRegression(random_state=2022, **param)
                        model.fit(train_x, train_y)
                        y_pred = model.predict(val_x)
                        kpi = oos_rsquare(val_y, y_pred)
                        #param check
                        if self.debug_mode:
                            cv_param_list = []
                            cv_param_list.append(combo)
                            cv_param_list.append(str(valid_train_start_date))
                            cv_param_list.append(str(valid_train_end_date)) 
                            cv_param_list.append(str(valid_end_date))
                            cv_param_list.append(kpi)
                        kpis.append(kpi)
                        if self.debug_mode:
                            self.cv_param_details = self.cv_param_details.append(pd.Series(cv_param_list, index = self.cv_param_details.columns), ignore_index=True)
                    combo_kpi[combo] = np.mean(kpi)
                return np.mean(kpis)

            study = optuna.create_study(
            direction="maximize", 
            sampler=self.sampler, 
            study_name=f'train_end_date_{self.train_end_date}'
            )

            study.optimize(objective,n_trials=self.backtest_config.tune_trails)
            # use best hyperparameters to train this part of back test
            param = study.best_params
            combo = max(combo_kpi, key=combo_kpi.get)
            print('best chosen backtest combination:', combo)
            
            #param_list = [] # for testing
            #param_list.append(combo)
            train_valid_period = combo[0]
            test_period = combo[1]
            train_period = combo[2]
            topk = combo[3]
            fs_tool = combo[4]
            valid_period = test_period
            # change year to month
            train_valid_period *= 12
            train_period *= 12

            y_col = f'{self.y_col_prefix}{test_period}'
            data = self.df[self.dummy_cols + self.feat_cols + [y_col]].copy()
            data = data.dropna()
            actual_train_end_date = self.train_end_date - relativedelta(months=test_period - 1)

            train_start_date = self.train_end_date - relativedelta(months=train_valid_period)
            #param check
            if train_start_date < pd.to_datetime('200001', format='%Y%m'):
                raise ValueError('invalid first_train_end_ym value')
            predict_end_date = self.train_end_date + relativedelta(months=test_period)
            predict_dates = pd.date_range(
                start=self.train_end_date + relativedelta(months=1), end=min(predict_end_date, self.df['date'].max()), freq='MS')

            valid_date_sets = rolling_valid_split(self.kfold, train_period, train_start_date, actual_train_end_date, valid_period)
            train_data = data.query(f'"{train_start_date}" < date <= "{actual_train_end_date}"')

            # balance data - downsampling
            if self.md.lower() == 'lr':
                # balance data - downsampling
                train_data = downsampling(train_data)
            #param_list.append(train_data.shape[0])
            X_train = train_data[self.feat_cols].values
            y_train = train_data[y_col].values

            # implement feature selection, 'mi', 'boruta', 'ae', 'rfecv', ''
            try:
                if self.backtest_config.paper_id.lower() == 'all':
                    _, select_ids = feature_selection(X_train, y_train, method=fs_tool, k=200)             
                else:
                    _, select_ids = feature_selection(X_train, y_train, method=fs_tool, k=25)         
            except:
                _, select_ids = feature_selection(X_train, y_train, method=fs_tool, k=5) 

            #_, select_ids = X_train[:,select_feature_idx], select_feature_idx
            if sum(select_ids) == 0:
                select_ids = [True for _ in self.feat_cols] 
                
            if self.backtest_config.md == 'lgb':
                model = LGBMRegressor(seed=2022, **param)
            elif self.backtest_config.md == 'lr':
                model = LogisticRegression(random_state=2022, **param)
            model.fit(X_train[:, select_ids], y_train)
            # record investment changes for next `test_period` months
            # run prediction in the first predict month and rebalance the investment after `test_period` month
            # print('--------\n', predict_dates, '\n--------\n')
            predict_date = predict_dates[0]
            test_data = data.query(f'date == "{predict_date}"')
            X_test = test_data[self.feat_cols].values[:, select_ids]
            y_test = test_data[y_col].values

            outputs = test_data[self.dummy_cols + [y_col]].copy()
            outputs[pred_col] = model.predict(X_test)
            #outputs['fut_ret'] = self.df_copy.query(f'date == "{predict_date}"')['fut_ret1']
            # descending sort values to select topk-stocks
            outputs = outputs.sort_values(by=[pred_col], ascending=False)
            #outputs = outputs.sort_values(by=[pred_col, 'fut_ret'], ascending=[False, False])
            # find out the legal ticker will have how many days recording during `test_period` months
            ideal_ser_num = sum([self.counting_max[(pdt.year, pdt.month)] for pdt in predict_dates])
            # equally separate investment to `topk` folds
            allocate_fund = init_fund / topk
            # store the portfolio value changing records during next `test_period` months
            ticker_recs_this_month = []
            # store the expected portfolio value for next rebalance strategy
            new_money = 0.
            idx, cnt = 0, 0
            tickers_tmp_info = []
            while cnt < topk:
                # select the ticker name with rank number `idx`
                ticker_name = outputs.iloc[idx, :]['ticker']
                # select the daily changing prices of this ticker
                et = predict_dates[-1] + pd.tseries.offsets.MonthEnd(n=1)
                try: 
                    sp500_historical_tickers = self.sp500_historical_list[(self.sp500_historical_list['date'].dt.year == pd.to_datetime(et).year) \
                                                & (self.sp500_historical_list['date'].dt.month == pd.to_datetime(et).month)].iloc[-1]['tickers']
                except:
                    pass
                ticker_ts = self.daily_prc.query(
                    f'"{str(predict_dates[0].date())}" <= date <= "{str(et.date())}" and ticker == "{ticker_name}"')
                if ticker_name not in sp500_historical_tickers:
                    pass
                if len(ticker_ts) != ideal_ser_num:
                    # this ticker is illegal, just skip and find next. 
                    # if this ticker is not in the historical_tciker, skip and find next
                    idx += 1
                    continue
                else:
                    # legal ticker, just sort by datetime again for insurance
                    ticker_ts = ticker_ts.sort_values(by=['date']).reset_index(drop=True) 
                    # the first price in ticker_ts will be the purchase price of this ticker
                    purchase_price = ticker_ts.head(1)['adjclose'].values[0]
                    purchase_amount = allocate_fund / purchase_price
                    # record daily price for this ticker, this list will store value changes for each top-k ticker
                    # it will be summed in next code to represent the portfolio changes
                    if sum(np.isnan(purchase_amount * ticker_ts['adjclose'].values)) > 0:
                        idx += 1
                        continue
                    ticker_recs_this_month.append(purchase_amount * ticker_ts['adjclose'].values)
                    # the last price in ticker_ts will be the selling price of this ticker
                    sell_price = ticker_ts.tail(1)['adjclose'].values[0]
                    sell_return = sell_price * purchase_amount
                    # add the return money of this ticker after `test_period` months
                    new_money += sell_return
                    tickers_tmp_info.append(ticker_name)
                    cnt += 1
                    idx += 1
            selected_pf_daily = pd.DataFrame({
                'date': ticker_ts['date'].values, 
                'val': np.sum(ticker_recs_this_month, axis=0)}) 
            t1, t2 = selected_pf_daily['date'].min(), selected_pf_daily['date'].max()
            y_num = (t2 - t1).days / 365
            # todo: add total return
            ar = round(np.power(selected_pf_daily['val'].iloc[-1] / selected_pf_daily['val'].iloc[0], 1 / y_num) - 1, 4)
            selected_pf_daily['cummax'] = selected_pf_daily['val'].cummax()
            selected_pf_daily['drawdown'] = (selected_pf_daily['val'] - selected_pf_daily['cummax']) / selected_pf_daily['cummax']
            mdd = round(selected_pf_daily['drawdown'].min(), 4)
            cr = ar / abs(mdd)

            calmar_ratio_res.loc[combo_cnt, :] = [train_valid_period//12, test_period, train_period//12, topk, fs_tool, cr]
            ar_res.loc[combo_cnt, :] = [train_valid_period//12, test_period, train_period//12, topk, fs_tool, ar]
            mdd_res.loc[combo_cnt, :] = [train_valid_period//12, test_period, train_period//12, topk, fs_tool, mdd]


            line = f'{str(t1.date())},{str(t2.date())},{train_valid_period//12},{test_period},{train_period//12},{topk},{fs_tool},{ar},{mdd},{cr}\n'
            f.write(line)
            f.flush()

            pf_daily_res[combo_cnt] = selected_pf_daily[['date', 'val']].copy()
            fund_res[combo_cnt] = new_money
            combo_cnt += 1

            best_combo = calmar_ratio_res.query(f'cr == {calmar_ratio_res.cr.max()}').index[0]
            best_test_perid = calmar_ratio_res.loc[best_combo, 'test_period']
            attach_pf_daily = pf_daily_res[best_combo]
            self.money = fund_res[best_combo]
            self.pf_daily_trend = pd.concat([self.pf_daily_trend, attach_pf_daily], ignore_index=True)

            # to next back test part
            self.train_end_date = self.train_end_date + relativedelta(months=best_test_perid)

            # debug
            if self.debug_mode:
                best_combo_chosen_list.append(combo) 
                best_combo_chosen_list.append(str(train_start_date)) 
                best_combo_chosen_list.append(str(actual_train_end_date))
                best_combo_chosen_list.append(str(predict_dates[0]))
                best_combo_chosen_list.append(sorted(combo_kpi, key=lambda t: t[1])[-1][1])
                self.best_combo_chosen_params = self.best_combo_chosen_params.append(pd.Series(best_combo_chosen_list, index = self.best_combo_chosen_params.columns), ignore_index=True)
            # finish recording, close the csv file
        f.close()
        if self.debug_mode:
            self.best_combo_chosen_params.to_csv(self.backtest_config.best_combo_chosen_params_path)
            self.cv_param_details.to_csv(self.backtest_config.cv_param_details_path)

    def plot(self):
        # output 3 kpi plots
        t1, t2 = self.pf_daily_trend['date'].min(), self.pf_daily_trend['date'].max()
        y_num = (t2 - t1).days /365

        spy_df = pd.read_csv(self.backtest_config.spy_daily_prc_path)
        spy_df['date'] = pd.to_datetime(spy_df['date'])
        spy_daily = spy_df.query(
            f'"{self.pf_daily_trend.date.min()}" <= date <= "{self.pf_daily_trend.date.max()}"')[['date', 'ticker', 'adjclose']]
        spy_daily['year'] = spy_daily.date.dt.year
        spy_daily['month'] = spy_daily.date.dt.month
        spy_daily = spy_daily.sort_values(by=['date']).reset_index(drop=True)  
        spy_daily['value'] = spy_daily['adjclose']

        self.pf_daily_trend['date'] = pd.to_datetime(self.pf_daily_trend['date'])
        self.pf_daily_trend['month'] = self.pf_daily_trend['date'].dt.month
        self.pf_daily_trend['year'] = self.pf_daily_trend['date'].dt.year
        self.pf_daily_trend.sort_values(['date'], inplace=True)
        self.pf_daily_trend['cummax'] = self.pf_daily_trend['val'].cummax()
        self.pf_daily_trend['drawdown'] = (self.pf_daily_trend['val'] - self.pf_daily_trend['cummax']) / self.pf_daily_trend['cummax']
        ar = round(np.power(self.pf_daily_trend['val'].iloc[-1] / self.pf_daily_trend['val'].iloc[0], 1 / y_num) - 1, 4) * 100

        yearly_return = self.pf_daily_trend.groupby(['year']).apply(
                    lambda x: (x['val'].iloc[-1] - x['val'].iloc[0]) / x['val'].iloc[0])
        max_drawdown = round(self.pf_daily_trend.drawdown.min(), 4) * 100 

        # make predit_dates for future plots
        tmp = self.pf_daily_trend[['year', 'month']].drop_duplicates()
        all_predict_dates = tmp['year'] * 100 + tmp['month']
        all_predict_dates = all_predict_dates.astype(str).agg(lambda x: pd.to_datetime(x, format='%Y%m')).values

        # calculate SPY related KPI
        spy_daily.sort_values(['date'], inplace=True)
        spy_money = self.backtest_config.money
        spy_buy_price = spy_daily.head(1)['adjclose'][0]
        spy_amt = spy_money / spy_buy_price
        spy_daily['value'] *= spy_amt

        spy_daily['cummax'] = spy_daily['value'].cummax()
        spy_daily['drawdown'] = (spy_daily['value'] - spy_daily['cummax']) / spy_daily['cummax']
        spy_ar = round(np.power(spy_daily['value'].iloc[-1] / spy_daily['value'].iloc[0], 1 / y_num) - 1, 4) * 100
        spy_max_drawdown = round(spy_daily.drawdown.min(), 4) * 100 
        spy_yearly_return = spy_daily.groupby(['year']).apply(
            lambda x: (x['value'].iloc[-1] - x['value'].iloc[0]) / x['value'].iloc[0])

        # Draw plots
        sns.set_theme(palette=sns.color_palette("Set1"))
        fig, ax = plt.subplots(nrows=3, ncols=1, figsize=(20, 15))
        fig.suptitle('Sample Paper 9. Forecasting daily stock market return using dimensionality reduction | Logistic | (201701, 201705)', \
                        fontsize=25, weight='bold')
        ax[0].plot(np.arange(len(self.pf_daily_trend['date'])), self.pf_daily_trend['val'], label='Porfolio')
        ax[0].plot(np.arange(len(spy_daily['date'])), spy_daily['value'], label='SPY')
        ax[0].axhline(self.backtest_config.money, ls='--', c='grey', alpha=0.5)

        ax[0].set_xlabel('Date', fontweight='bold', fontsize=18)
        ax[0].set_ylabel('Portfolio Value', fontweight='bold', fontsize=18)
        ax[0].text(x=0, y=1.16, s='Model vs SPY', 
                fontsize=22, weight='bold', ha='left', va='bottom', transform=ax[0].transAxes)
        ax[0].text(x=0., y=1.08, 
                s=f'Portfolio: Annual Return = {ar:.2f}%, Max DrawDown = {max_drawdown:.2f}%, Calmar Ratio = {ar/abs(max_drawdown):.2f};', 
                fontsize=15, alpha=0.8, ha='left', va='bottom', transform=ax[0].transAxes)
        ax[0].text(
            x=0., y=1.01, 
            s=f'SPY: Annual Return = {spy_ar:.2f}%, Max DrawDown = {spy_max_drawdown:.2f}%, Calmar Ratio = {spy_ar/abs(spy_max_drawdown):.2f}', 
            fontsize=15, alpha=0.8, ha='left', va='bottom', transform=ax[0].transAxes)

        y_scaler = np.linspace(min(self.pf_daily_trend['val'].min(), spy_daily['value'].min()), 
                            self.pf_daily_trend['val'].max(), 6)
        ax[0].set_yticks(y_scaler)
        ax[0].set_xticks(np.arange(0, len(self.pf_daily_trend['date']), 90))
        ax[0].set_xticklabels(
            [str(self.pf_daily_trend['date'].values[idx])[:10] for idx in np.arange(0, len(self.pf_daily_trend['date']), 90)])
        ax[0].legend(fontsize=16)
        # ax[0].xaxis.grid(False)

        barWidth = 0.2
        br1 = np.arange(len(yearly_return))
        ax[1].bar(br1 - barWidth / 2, yearly_return, width=barWidth, edgecolor='none', color='red', label='Porfolio (>0)')
        ax[1].bar(br1 + barWidth / 2, spy_yearly_return, width=barWidth, edgecolor='none', color='salmon', label='SPY (>0)')
        ax[1].bar(
            br1 - barWidth / 2, 
            [t if t <= 0 else 0 for t in yearly_return], 
            width=barWidth, edgecolor='none', color='forestgreen', label='Porfolio (<0)')
        ax[1].bar(
            br1 + barWidth / 2, 
            [t if t <= 0 else 0 for t in spy_yearly_return], 
            width=barWidth, edgecolor='none', color='limegreen', label='SPY (<0)')
        ax[1].set_xlabel('Year', fontweight='bold', fontsize=18)
        ax[1].set_ylabel('Yealy Return', fontweight='bold', fontsize=18)
        ax[1].set_xticks(br1)
        ax[1].set_xticklabels([idx for idx in yearly_return.index], fontsize=13)
        ax[1].set_title('Yearly Return (Model vs SPY)', fontweight='bold', fontsize=22, loc='left')
        y_scaler = np.linspace(min(yearly_return.min(), spy_yearly_return.min(), 0), max(yearly_return.max(), spy_yearly_return.max()), 6)
        ax[1].axhline(y=0, ls='--', color='gray', alpha=0.5)
        ax[1].set_yticks(y_scaler)
        ax[1].set_yticklabels([f'{round(v * 100)}%' for v in y_scaler])
        ax[1].legend(fontsize=16)

        ax[2].plot(
            np.arange(len(self.pf_daily_trend['date'])), self.pf_daily_trend['drawdown'], color='teal', label='Porfolio')
        ax[2].plot(
            np.arange(len(spy_daily['date'])), spy_daily['drawdown'], color='limegreen', label='SPY')
        ax[2].set_xlabel('Date', fontweight='bold', fontsize=18)
        ax[2].set_ylabel('Drawdown', fontweight='bold', fontsize=18)
        ax[2].set_title(f'Daily Drawdown (Model vs SPY)', fontweight='bold', fontsize=22, loc='left')
        ax[2].set_xticks(np.arange(0, len(self.pf_daily_trend['date']), 90))
        ax[2].set_xticklabels(
            [str(self.pf_daily_trend['date'].values[idx])[:10] for idx in np.arange(0, len(self.pf_daily_trend['date']), 90)])
        y_scaler = np.linspace(
            min(self.pf_daily_trend['drawdown'].min(), spy_daily['drawdown'].min()), 
            max(0, self.pf_daily_trend['drawdown'].max(), spy_daily['drawdown'].max()), 6)
        ax[2].axhline(y=0, ls='--', color='gray', alpha=0.5)
        ax[2].set_yticks(y_scaler)
        ax[2].set_yticklabels([f'{round(v * 100)}%' for v in y_scaler])
        ax[2].legend(fontsize=16)

        plt.tight_layout()
        plt.savefig(self.backtest_config.get_kpi_result_path(), format='png')
        plt.show()
        plt.close()

if __name__ == "__main__":
    start = time.time()
    backtest = BackTest()
    print('Currently [Logistic Regression Sample of', backtest.backtest_config.paper_id.lower(), '] is processing')
    backtest.train()
    backtest.plot()
    print('Please check result sample folders for results.')
    end = time.time()
    hours, rem = divmod(end-start, 3600)
    minutes, seconds = divmod(rem, 60)
    print("\nExecution completed in {:0>2}:{:0>2}:{:05.2f}".format(
        int(hours), int(minutes), seconds))
