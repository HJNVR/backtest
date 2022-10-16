## Overview

This project aims to build machine learning model for prediction of future stock performance. The model will use features generated from 11 academic papers to predict future stock prices and select best-performing stocks from the S&P500 index. The selected stocks will form a potfolio whose performance will then be compared against the performance of the S&P500 index.

The methodology used in this project include:

 - Highly robust feature selection and leak detection
 - Accurate and various hyper-parameter optimization in high-dimensional space
 - State-of-the-art predictive models for regression task about rate of return
 - Automatic results evaluation with metrics reports and plots
 - Predictions with interpretations

## Structure of Repository folders/files

 - **`config`**: All configuration json files are stored here
 - **`data`**: This is the warehouse to store all required or automatically generated datasets for the experiments and the whole project
 - **`high-priority-papers`**: Feature generation from reference papers shoudlb be done in this folder for future usage and referring
 - **`result`**: An automatically generated folder used to keep all results that generated by the code
    - **sample**: For users to compare their results with ours to make sure users are running the correct way
 - **`src`**: This is the source folder where main functions are
    - **`tools`**: It contains all codes with well-wrapped functions and methods that will be frequently used in the future

## AWS setting 
 - Register AWS account and genreate new access secret key
 - Install AWS CLI 
    - in the terminal, run: `$ pip install awscli`
      - for windows : `https://awscli.amazonaws.com/AWSCLIV2.msi`
    - in the terminal, run: `$ aws configure`
    - AWS Access Key ID: AKIAVNMK4EB****
    - AWS Secret Access Key: KZXHnAhU5Sd0sR5F****
    - Default region name: ap-southeast-1
    - Default output format: json
 - Go to aws folder
    - in the terminal, run: `$ cd .aws`
    - in the terminal, run: `$ cat credentials`
- Install boto3
    - in the terminal, run: `$ pip install boto3`
- check `status.csv`
    - search `S3` in AWS website and find `backtest` bucket

## Dependencies

The basic requirements are listed below:

```
boruta==0.3
yahoo_fin==0.8.9.1
lightgbm==3.3.1
python-dateutil==2.8.1
pandas==1.1.3
numpy==1.19.2
scikit-learn
tensorflow==2.7.0
keras==2.7.0
tqdm==4.50.2
optuna==2.10.0
```

More Details about all the packages dependencies for my Mac laptop are recorded in `requirements.txt`. Check it if you have any unexpected version issue.

## Configuration Settings

Take the configuration of back test as an example:
Definitions of configurations: 
- ml_parameters : parameters for Machine Learning models
    - `kfold` : split `train_valid_period` into k subperiods and all subperiods have the same `train_period` length
    - `train_period` : training period for one subperiod of `kfold` splitting
    - `topk` : select top-k stocks with highest prices
    - `money` : initial investing fund 
    - `first_train_end_ym` : the beginning timedate of entire backtesting 
    - `end_date` : the ending timedate of entire backtetsing
    - `tune_sampler` : algorithms to optimize hyperparameters. tpe: Tree-Structured Parzen Estimarot(TPE)
    - `train_valid_perios` : the actual trainning periods
    - `tune_trials` : the number of trials for each optimization process. 
    - `md` : ML models
    - `fs` : feature selection algorithms
    - `y_col_prefix` : target variables have prefix 'fut_ret'
    - `primary_key` : key info of each stock
- `logisitc_parameters` : 
    - `percentile` : rank and obtain stocks with the top 20% future return
    -  `threshold` : convert future return to 1, if future return >= threshold. Otherwise, convert future return to -1, if future return < threshold.
- `common_settings` :  
    - `paper_id` : the unique id for each paper ['all', 'paper1', 'paper3', 'paper4', 'paper6', 'paper7', 'paper9', 'paper11']
    - `debug_mode` : turn on debug mode if T. turn off degug mode if F. 
    - `parallel_compute` : turn on parallel computing if T. turn off parallel computing if F. 
- `file_paths` : file paths for papers
```
{   "ml_config": "Below are config settings for ML model",
    "ml_parameters" : {
                "kfold": 5,
                "train_period": [10, 15],
                "test_period": [1], 
                "topk": [10],
                "money": 100000, 
                "comment": "first_train_end_ym should be at least 2000 + the max of train_valid_period",
                "first_train_end_ym": "201701",
                "comment" : "end_date cannot beyond the last date of dataframe", 
                "end_date": "202203",
                "tune_sampler": "tpe",
                "train_valid_period": [16, 17],
                "tune_trials": 1,
                "method_comment" : "logisitc regression, xgboost",
                "md": "lr",
                "fs_commnent" : "available feature selection methods: lgb, boruta, empty, rfecv, mi",
                "fs": ["mi", ""],
                "y_col_prefix": "fut_ret",
                "primary_key": ["ticker", "permno", "date"]

                },
    "logistic_parameters" : {
                "percentile": 0.2,
                "threshold": 0.1
                },
    "common_settings" :{
                "paper_id_comment" : "choose from ['all', 'paper1', 'paper3', 'paper4', 'paper6', 'paper7', 'paper9', 'paper11', 'sample']",
                "paper_id" : "paper9",
                "debug_mode_comment" : "T/F",
                "debug_mode" : "F",
                "parallel_compute_comment" : "T/F number of virtual machine",
                "parallel_compute" : "F"
                },
    "file_paths": {
                "result_path" : "../result/",
                "sp500_historical_list_path" : "../data/sp500_historical_list.csv",
                "spy_daily_prc_path" : "../data/spy_daily_prc.csv",
                "sp500_daily_prc_path" : "../data/sp500_daily_prc.csv",
                "paper1_features_path" : "../data/paper1_features.csv",
                "paper3_features_path" : "../data/paper3_features.csv",
                "paper4_features_path" : "../data/paper4_features.csv",
                "paper6_features_path" : "../data/paper6_features.csv",
                "paper7_features_path" : "../data/paper7_features.csv",
                "paper9_features_path" : "../data/paper9_features.csv",
                "paper11_features_path" : "../data/paper11_features.csv",
                "consolidated_features_path" : "../data/consolidated_features.csv",
                "paper1_final_summary_path" : "../result/paper1/final_summary.csv",
                "paper3_final_summary_path" : "../result/paper3/final_summary.csv",
                "paper4_final_summary_path" : "../result/paper4/final_summary.csv",
                "paper6_final_summary_path" : "../result/paper6/final_summary.csv",
                "paper7_final_summary_path" : "../result/paper7/final_summary.csv",
                "paper9_final_summary_path" : "../result/paper9/final_summary.csv",
                "paper11_final_summary_path" : "../result/paper11/final_summary.csv",
                "consolidated_final_summary_path" : "../result/consolidated/final_summary.csv",
                "lr_sample_final_summary_path" : "../result/sample/lr_sample_final_summary.csv",
                "lgb_sample_final_summary_path" : "../result/sample/lgb_sample_final_summary.csv",
                "paper1_kpi_result_path" : "../result/paper1/backtest_kpi.png",
                "paper3_kpi_result_path" : "../result/paper3/backtest_kpi.png",
                "paper4_kpi_result_path" : "../result/paper4/backtest_kpi.png",
                "paper6_kpi_result_path" : "../result/paper6/backtest_kpi.png",
                "paper7_kpi_result_path" : "../result/paper7/backtest_kpi.png",
                "paper9_kpi_result_path" : "../result/paper9/backtest_kpi.png",
                "paper11_kpi_result_path" : "../result/paper11/backtest_kpi.png",
                "consolidated_kpi_result_path" : "../result/consolidated/backtest_kpi.png",
                "lr_sample_kpi_result_path" : "../result/sample/lr_backtest_kpi.png",
                "lgb_sample_kpi_result_path" : "../result/sample/lgb_backtest_kpi.png",
                "best_combo_chosen_params_path" : "../result/best_combo_chosen_params.csv",
                "cv_param_details_path" : "../result/cv_param_details.csv"
                }
}

```

## How to Run

- Step 1 : Set up virtual environment (`backtest_env`)
    - Go to root directory -> `$ conda env create -f env.yml`
    - In the same directory -> `$ conda activate backtest_env`
    - check location of 'backtest_env' -> `$ conda backtest_env list`

- Step 2 : Download the latest spy and spy market data from yahoo-finance
    - Go to `./src/tools/yh_data.py` -> `$ python yh_data.py`
        - Generate `sp500_dily_prc.csv` and `spy_daily_prc` in the same directory
        ```
        df = download_sp500("1999-12-01", "2022-03-31")
        df.to_csv('./sp500_daily_prc.csv', index=False)
        df = download_spy("1999-12-01", "2022-03-31")
        df.to_csv('./spy_daily_prc.csv', index=False)
        ```
    - SP500_list can be downloaded here: https://github.com/fja05680/sp500/blob/master/S%26P%20500%20Historical%20Components%20%26%20Changes(08-12-2022).csv
    - Make sure all files are in `./data` folder

- Step 3 : Feature generation of selected papers
    - Go to `./high-priorty-papers/src` -> `$ python generate_all_features.py`
    - Make sure all files are in `./data` folder

- Step 4: The main file will be in the `./src/backtest.py`.
    - Can test with sample files firstly
        - `$ python backtest_lr_sample.py`
        - `$ python backtest_lgb_sample.py`
    - Finally, after modifying the information in the correspoding configuration file in `config`, run:
    ```
    python backtest.py
    ```
    After completion, all expected results will wait for you in the `result` folder.


## Outputs
structure(one output for each paper(1-11), all combine totger one output), one output three charts and a summary.csv for each folder  

- Sample folder: contains two sample results for both Logisitc Regression and LightGBM
    - Go to `./config/back_test_sample_lr.json` and `back_test_sample_lgb.json` for config setting details
    - Go to `./src/backtest_lr_sample.py` and `./src/backtest_lgb_sample.py` for codes details

Considering different combinations of `train_valid_period` and `test_period`, in the `result` folder there will be different folders with name rules `A_B`, a spreadsheet to record the corresponding metrics for these combinations, like:

| **train_valid_period** | **test_period** | **Calmar Ratio** | **annual return** | **max draw down** |
|------------------------|-----------------|-------------------|-------------------| ---|
| 11                      |      1           |       0.419961679            |          24.11         |-57.41 |
		

## TODO List
- To allow features with daily frequency to be fed into the ml model
- To include stocks in the Nasdaq and other widely used indices in our model
- Multidimensional heatmap
