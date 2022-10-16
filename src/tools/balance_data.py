import pandas as pd

def downsampling(train_data):
    # to do: test_period > 1
    if train_data[train_data['fut_ret1']== -1].shape[0] >= train_data[train_data['fut_ret1']== 1].shape[0]:
        sample = train_data[train_data['fut_ret1'] == -1].sample(train_data[train_data['fut_ret1']== 1].shape[0], random_state=2022)
        train_data = pd.concat([train_data[train_data['fut_ret1'] == 1], sample])
    else:
        sample = train_data[train_data['fut_ret1'] == 1].sample(train_data[train_data['fut_ret1']== -1].shape[0], random_state=2022)
        train_data = pd.concat([train_data[train_data['fut_ret1'] == -1], sample])
    return train_data