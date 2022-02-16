import pandas as pd
import numpy as np
import sys, os
import logging
from dotenv import load_dotenv
load_dotenv(verbose=False)
sys.path.append(os.getenv("ROOT_DIR"))

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import SMOTE
from sklearn.impute import SimpleImputer
from sklearn.compose import make_column_selector as selector
from ja_pk.utils import basic


def get_preprocessing_pipe(filename, feature_data, target_data):

    pipe_config = []
    try:
        pipe_config = basic.read_config(filename)
        logging.info(f' Preprocess pipeline to run: {pipe_config}')
    except:
        logging.info('No preprocess steps were defined in the preprocess yml file')


    preprocess_tuple_list = []
    features_resambled = "none"
    target_resambled = "none"
  
    if len(pipe_config) > 0:
        for pipe_obj in pipe_config:

            # retrieve the function name of the step
            function = pipe_config.get(pipe_obj).func
            args = dict(pipe_config.get(pipe_obj))

            if function == 'pca':
                pca = PCA()
                preprocess_tuple_list.append(('pca', pca))
            
            if function == 'minmaxscaler':
                if len(args) == 1:
                    mmsc = MinMaxScaler()
                else:
                    args.pop('func')
                    mmsc = MinMaxScaler(**args)
                preprocess_tuple_list.append(('mmsc', mmsc))

            if function == 'onehotencoding':
                cols = args['cols']
                for col in cols:
                    ohe = ColumnTransformer([(col, OneHotEncoder(categories='auto'), [-1])], remainder = 'passthrough')
                    preprocess_tuple_list.append((col+'_ohe', ohe))

            if function == 'SMOTE':
                # rebalance
                features_resambled, target_resambled = SMOTE().fit_resample(feature_data, target_data)
                
            if function == 'simpleimpute':
                imp = SimpleImputer(missing_values=np.nan, strategy='median')
                imp.fit(feature_data)
                features_resambled = imp.transform(feature_data)

            #ohe = ColumnTransformer([("dim_time_month_new", OneHotEncoder(), [-1])], remainder = 'passthrough')
            #pca = PCA()

            #steps = [('pca', pca), (self.modelname, self.model)]
            #steps = [('std', std), (self.modelname, self.model)]
            #steps = [('mmsc', mmsc), ('ohe', ohe), (self.modelname, self.model)]
            #steps = [('mmsc', mmsc), (self.modelname, self.model)]
            #steps = [(self.modelname, self.model)]
    return preprocess_tuple_list, features_resambled, target_resambled
