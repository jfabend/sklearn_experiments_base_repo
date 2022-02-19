import logging
logging.basicConfig(level = logging.INFO)
from sklearn import model_selection as ms
#from sklearn.pipeline import make_pipeline
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_validate 
from ja_pk.exp.preprocess import get_preprocessing_pipe
import pandas as pd

class Experiment():

    def __init__(self, data, feature_list, target_col, model, modelname, param_grid):
        """[summary]

        Args:
            data (dataframe): dataframe including the data for train and test
            feature_list (list): list of features
            target_col (str): name of target column
            model (Object): Model Object created with sklearn
            param_grid (dict): Dict or list of dicts including the model params names as keys and a list of values as value

        Returns:
            Object: Object of class Experiment
        """
        self.data = data
        self.feature_list = feature_list
        self.target_col = target_col
        self.feature_data = data[feature_list]
        self.target_data = data[target_col]

        self.model = model
        self.modelname = modelname
        self.param_grid = param_grid
        return None

# Perfom gridsearch cross validation and evaluation
    def start(self):
        print("Feature Data Sample:")
        print(self.feature_data.sample(10))
        
        # initialize scikit model tuple
        model_tuple = (self.modelname, self.model)

        # initiailze preprocess tuple and append it to the model tuple
        # preprocessing could also include resampling / balancing of the data
        # hence, we need to hand over the datasets to the preprocessing function
        preprocessfile = "\\exp\\preprocess_pipe.yml"
        preprocess_tuples, features_resambled, target_resambled  = get_preprocessing_pipe(preprocessfile, self.feature_data, self.target_data)
        preprocess_tuples.append(model_tuple)
        pipe = Pipeline(steps = preprocess_tuples)

        # If the data was resampled during preprocessing
        # overwrite self.feature_data and self.target_data
        if type(features_resambled) != str:
            self.feature_data = features_resambled

        if type(target_resambled) != str:
            self.target_data = target_resambled
            print("New balance of the target classes:")
            print(target_resambled.value_counts())

        # Cross Validation Parameters
        # Move this to the exp_config.yml

        my_scoring = ['neg_root_mean_squared_error' , 'neg_mean_absolute_percentage_error']
        #my_scoring = 'roc_auc'
        # my_scoring = ['accuracy', 'roc_auc'] => !! Das geht nicht !!
        folds = 5

        # if there is no param grid, start the simple scikit cross_validate()
        if self.param_grid == "none":
            results = cross_validate(pipe, self.feature_data, self.target_data, cv=folds, scoring=my_scoring)

        # if there is a param grid, start scikit GridSearchCV()    
        else:
            logging.info(f'Building the GridSearchCV')
            grid = ms.GridSearchCV(pipe, self.param_grid, cv=folds, scoring =my_scoring, return_train_score=False)
            logging.info(f'Starting the fitting process')
            grid.fit(self.feature_data, self.target_data)
            results = pd.DataFrame(grid.cv_results_)[['mean_test_score', 'std_test_score', 'params']] 
            print(grid.best_params_)
        return results