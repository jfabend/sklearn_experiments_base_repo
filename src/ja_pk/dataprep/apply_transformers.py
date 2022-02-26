from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import PowerTransformer
import pandas as pd
from ja_pk.dataprep import prep_funcs

# load the two data sets
data = pd.read_csv("train_prepped.csv")
test_set = pd.read_csv("test_prepped.csv")
trans_feature_list = ['account_balance']

# Fit transformers on the train set and apply them there
#scaler = MinMaxScaler()
scaler = StandardScaler()
data[trans_feature_list] = scaler.fit_transform(data[trans_feature_list])

# Apply the transformer also on the test set
test_set[trans_feature_list] = scaler.transform(test_set[trans_feature_list])

data.to_csv('train_prepped.csv', index=False)
test_set.to_csv('test_prepped.csv', index=False)