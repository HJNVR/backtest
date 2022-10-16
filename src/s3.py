import boto3
import pandas as pd
import io

def upload_file():
	s3 = boto3.client('s3')
	s3.upload_file('status.csv', 'mas-backtest', 'status.csv')

def read_file():
	s3 = boto3.resource('s3')
	#,
    #aws_access_key_id= 'YOUR_ACCESS_KEY_ID',
    #aws_secret_access_key='YOUR_SECRET_ACCESS_KEY')
	obj = s3.Object('mas-backtest', 'status.csv')
	data=obj.get()['Body'].read()
	status = pd.read_csv(io.BytesIO(data), header=0, delimiter=",", low_memory=False, index_col = 0)
	return status

upload_file()
#print(read_file())