from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import PowerTransformer
from sklearn.manifold import LocallyLinearEmbedding
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

# Embedding
# embedd_features =    [
#     'number_of_products_1',
#     'number_of_products_2',
#     'number_of_products_3',
#     'number_of_products_4',
#     'has_life_insurance_0',
#     'has_life_insurance_1']

# n_components = 3
# embedding = LocallyLinearEmbedding(n_components=n_components, eigen_solver='dense')
# embedded_train_array = embedding.fit_transform(data[embedd_features])
# embedded_test_array = embedding.transform(test_set[embedd_features])

# embedd_cols = [f'embedd_{i}' for i in range(0, n_components)]
# embedded_train_df = pd.DataFrame(columns=embedd_cols, data=embedded_train_array)
# embedded_test_df = pd.DataFrame(columns=embedd_cols, data=embedded_test_array)

# data.reset_index(drop=True, inplace=True)
# test_set.reset_index(drop=True, inplace=True)
# embedded_train_df.reset_index(drop=True, inplace=True)
# embedded_test_df.reset_index(drop=True, inplace=True)

# data = pd.concat([data, embedded_train_df], axis=1)
# test_set = pd.concat([test_set, embedded_test_df], axis=1)

data.to_csv('train_prepped.csv', index=False)
test_set.to_csv('test_prepped.csv', index=False)