import os 
import json 
import boto3
import shutil
import pandas as pd


with open('../config/back_test.json') as config_file:
    data = json.load(config_file)

d = {}
count = 1 
for i in range(int(data['first_train_end_ym'][:4]), int(data['end_date'][:4])+1):
    for j in range(12):
        if len(str(j+1)) >= 2:
            d[count] = str(i)+str(j+1)
        else:
            d[count] = str(i)+'0'+str(j+1)
        count += 1 

l = [] # a list of date keys to be removed
for key, val in d.items():
    if key < [k for k, v in d.items() if v == data['first_train_end_ym']][0] or \
        key > [k for k, v in d.items() if v == data['end_date']][0]:
        l.append(key)

for i in l:
    del d[i]

train_end_l = []
end_date_l = []

for key in list(d.keys())[:-1]:
    train_end_l.append(d[key])
    end_date_l.append(str(int(d[key+1])))

# generate status.csv file

status = pd.DataFrame()
status['task_id'] = range(1, len(train_end_l)+1)
status['task_status'] = 'P'
status['first_train_end_ym'] = train_end_l
status['end_date'] = end_date_l
status['start_time_date'] = None
status['end_time_date'] = None
status['execution_time'] = None
#status['health_condition'] = None
status['heart_beat_time'] = None
status.to_csv('status.csv')

def upload_file():
    s3 = boto3.client('s3')
    s3.upload_file('status.csv', 'backtest-11-papers', 'status.csv')

upload_file()