from dateutil.relativedelta import relativedelta

# rolling window validation
def rolling_valid_split(
    kfold=5, train_period=None, train_start_date=None, train_end_date=None, valid_period=None):
    valid_date_sets = []

    t1 = train_start_date
    t2 = train_end_date - relativedelta(months=train_period + valid_period)
    delta = (t2.year - t1.year) * 12 + t2.month - t1.month
    window_size = int(delta / kfold)
    for k in range(kfold):
        if k == kfold - 1: # the last kfold calculate backwards 
            valid_end_date = train_end_date
            valid_train_end_date = valid_end_date - relativedelta(months=valid_period)
            valid_train_start_date = valid_train_end_date- relativedelta(months=train_period)
        else:
            valid_train_start_date = t1 + relativedelta(months=window_size * k)
            valid_train_end_date = valid_train_start_date + relativedelta(months=train_period)
            valid_end_date = valid_train_end_date + relativedelta(months=valid_period)
        valid_date_sets.append((valid_train_start_date, valid_train_end_date, valid_end_date))
    
    return valid_date_sets


# expanding window validation
def expanding_valid_split(
    kfold=5, train_period=None, train_start_date=None, train_end_date=None, valid_period=None):
    valid_date_sets = []
    t1 = train_start_date + relativedelta(months=train_period)
    t2 = train_end_date - relativedelta(months=valid_period)
    delta = (t2.year - t1.year) * 12 + t2.month - t1.month
    window_size = int(delta / (kfold - 1)) if kfold != 1 else 0
    for k in range(kfold):
        valid_train_start_date = train_start_date
        if k == kfold - 1:
            valid_train_end_date = t2
            valid_end_date = train_end_date
        else:
            valid_train_end_date = t1 + relativedelta(months=window_size * k)
            valid_end_date = valid_train_end_date + relativedelta(months=valid_period)   

        valid_date_sets.append((valid_train_start_date, valid_train_end_date, valid_end_date)) 
        
    return valid_date_sets