# load SPY and SPY markte data
# load corresponding feature data for specific paper_id
# load S&P 500 historical data

# ============================================================================

import pandas as pd
import numpy as np
from backtest_config import BacktestConfig

class BacktestDataset:
	def __init__(self, lr_sample=False, lgb_sample=False):
		self.backtest_config = BacktestConfig(lr_sample, lgb_sample)

	def get_daily_prc(self):
		# this data can be automatically generated by yh_data.py in tools
		daily_prc = pd.read_csv(self.backtest_config.sp500_daily_prc_path)
		daily_prc['date'] = pd.to_datetime(daily_prc['date'])
		daily_prc['year'] = daily_prc['date'].dt.year
		daily_prc['month'] = daily_prc['date'].dt.month
		daily_prc = daily_prc.sort_values(['ticker', 'date']).reset_index(drop=True)
		daily_prc = daily_prc.fillna(0)
		return daily_prc

	def get_sp500_historical_list(self):
		df = pd.read_csv(self.backtest_config.sp500_historical_list_path)
		df['date'] = pd.to_datetime(df['date'])
		return df


	def get_paper_features(self):
		if self.backtest_config.paper_id.lower() == 'all':
			df = pd.read_csv(self.backtest_config.consolidated_features_path, index_col = 0)
		elif self.backtest_config.paper_id.lower() == 'paper1':
			df = pd.read_csv(self.backtest_config.paper1_features_path, index_col = 0)
		elif self.backtest_config.paper_id.lower() == 'paper3':
			df = pd.read_csv(self.backtest_config.paper3_features_path, index_col = 0)
		elif self.backtest_config.paper_id.lower() == 'paper4':
			df = pd.read_csv(self.backtest_config.paper4_features_path, index_col = 0)
		elif self.backtest_config.paper_id.lower() == 'paper6':
			df = pd.read_csv(self.backtest_config.paper6_features_path, index_col = 0)
		elif self.backtest_config.paper_id.lower() == 'paper7':
			df = pd.read_csv(self.backtest_config.paper7_features_path, index_col = 0)
		elif self.backtest_config.paper_id.lower() == 'paper9':
			df = pd.read_csv(self.backtest_config.paper9_features_path, index_col = 0)
		elif self.backtest_config.paper_id.lower() == 'paper11':
			df = pd.read_csv(self.backtest_config.paper11_features_path, index_col = 0)
		elif self.backtest_config.paper_id.lower() == 'sample': # sample will use paper9
			df = pd.read_csv(self.backtest_config.paper9_features_path, index_col = 0)
		else:
			raise ValueError('invalid paper id')

		# processing the raw data
		df['date'] = pd.to_datetime(df['date'])
		# rename the `date` in origin data to `trade_date`
		df['trade_date'] = df['date'].values
		y_cols = [f'fut_ret{col}' for col in self.backtest_config.test_period]
		dummy_cols = self.backtest_config.dummy_cols + ['trade_date']
		feat_cols = [col for col in df.columns if col not in (dummy_cols + y_cols)]
		# new `date` will be used to trace the back test process
		df['date'] = df['date'] + pd.tseries.offsets.MonthEnd(n=0) - pd.tseries.offsets.MonthBegin(n=1)
		df.replace([np.inf, -np.inf], np.nan, inplace=True)
		#df[feat_cols] = df[feat_cols].fillna(0)
		df = df.fillna(0)
		return df

if __name__ == "__main__":
	backtest_dataset = BacktestDataset()
	#print(backtest_dataset.get_sp500_historical_list())
	print(backtest_dataset.get_paper_features())