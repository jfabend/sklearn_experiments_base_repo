import sys, getopt, os
import pickle
from joblib import dump, load
from dotenv import load_dotenv
load_dotenv(verbose=False)
import logging
logging.basicConfig(level = logging.INFO)
#sys.path.append(os.getenv("ROOT_DIR"))

#from db.config import Config
#from db.connect_db import DbConnection
#from db.get_dbtable_data import get_dbtable_data
from ja_pk.exp.model import Model
from ja_pk.exp.experiment import Experiment

import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import r2_score
from sklearn.metrics import median_absolute_error

# Read exp config yml
import yaml
from box import Box
with open(os.getenv("ROOT_DIR") + "\\exp\\exp_config.yml", "r") as ymlfile:
  exp_config = Box(yaml.safe_load(ymlfile))

# Read table from DB (pandas df)
#! data = get_dbtable_data(exp_config.table_name)
data = pd.read_csv("train_prepped.csv")

# Read features and target
feature_list = exp_config.feature_list
target = exp_config.target

# Prepair an empty list for all models to be applied
models_to_apply = []

# Check if models contains only one modelname
if type(exp_config.models) is str:
  modelname = exp_config.models

  # Initialize model object according to given model name
  # without any params (without further info, default values would be used)
  next_model_class = Model(model_name = modelname)
  next_model_object = next_model_class.return_model()
  models_to_apply.append(next_model_object)

  # Create empty param grid
  model_params_raw = "none"

  experiment = Experiment(data=data,
                          feature_list=feature_list,
                          target_col=target,
                          model=models_to_apply[0],
                          modelname=modelname,
                          param_grid=model_params_raw)
  results = experiment.start()

# ... or if the models contain models with pararms
else:
  for modelname in exp_config.models:

      # Initialize model  object according to given model name
      # without any params (without further info, default values would be used)
      next_model_class = Model(model_name = modelname)
      next_model_object = next_model_class.return_model()
      models_to_apply.append(next_model_object)

      # create param grid for current model
      model_params_raw = exp_config.models.get(modelname)

      experiment = Experiment(data=data,
                              feature_list=feature_list,
                              target_col=target,
                              model=models_to_apply[0],
                              modelname=modelname,
                              param_grid=model_params_raw)

  logging.info(f'Starting the experiment now')
  results = experiment.start()

print(results)
model = results['estimator'][0][0]
dump(model, 'model.joblib')
test_set = pd.read_csv("test_prepped.csv")
test_set['predictions'] = model.predict(test_set[list(experiment.feature_list)])

mape = mean_absolute_percentage_error(test_set[experiment.target_col], test_set['predictions'])
rsme = mean_squared_error(test_set[experiment.target_col], test_set['predictions'], squared=False)
r2 = r2_score(test_set[experiment.target_col], test_set['predictions'])
median_ae = median_absolute_error(test_set[experiment.target_col], test_set['predictions'])

print(f'mape: {mape}')
print(f'rsme: {rsme}')
print(f'r2: {r2}')
print(f'median_ae: {median_ae}')