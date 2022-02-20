import pandas as pd
import sys, os
import logging
from dotenv import load_dotenv
load_dotenv(verbose=False)
sys.path.append(os.getenv("ROOT_DIR"))

#from db.get_dbtable_data import get_dbtable_data
#from db.write_table import write_table
from ja_pk.utils import basic
from ja_pk.dataprep import prep_funcs

logging.basicConfig(level = logging.INFO)

# Read the pipeline config
pipe_config_pipe = "\\dataprep\\data_pipe_oe.yml"
pipe_config = basic.read_config(pipe_config_pipe)
#! data = get_dbtable_data(db_table_name)
#data = pd.read_csv("C:\Data\projects\kaggle\\titanic\\train.csv")
data = pd.read_csv("C:\Data\projects\\nomoko_ass\\immo_data.csv",
                    sep=',',
                    encoding = 'utf8',
                    lineterminator='\n'
                )
data = data.sample(frac = 0.2)

logging.info(f' Prep pipeline to run: {pipe_config_pipe}')
#logging.info(f' Database table used for prep pipeline: {db_table_name}')

# run the pipeline
def run_pipeline(df):
    """runs a dataprep pipeline consisting of a sequence of several functions transforming a given dataframe.

    This function reads a yml-File and identifies the different function components of the pipeline.

    Each component in the yml-File includes the name of the function which is defined in dataprep.py.
    In addition to that, the component definition in the yml-File contains also the arguments which
    should be used when calling the funtion.

    Afterwards, it loops through this components and calls the given functions using the given arguments.

    Arguments:
        df {pandas dateframe} -- Pandas dataframe

    Returns:
        A dataframe
    """

    tmp_df = df

    # Loop through each steps (pipe_obj) of the pipeline
    for pipe_obj in pipe_config:

        logging.info(f' Pipe Step: {pipe_obj}')

        # retrieve the function name of the step
        function = pipe_config.get(pipe_obj).func
        args = dict(pipe_config.get(pipe_obj))

        # retrieve the arguments for the function call
        if len(args) == 1:
            print("no arguments")
            args = None
        else:
            del args['func']
            print(args)
        
        # parse the function and call it using the given arguments
        target_func = getattr(prep_funcs, function)
        if args:
            tmp_df = tmp_df.pipe(target_func, **args)
            #print(tmp_df)
        else:
            tmp_df = tmp_df.pipe(target_func)
            #print(tmp_df)
    return tmp_df

new_df = run_pipeline(data)
new_df.head()
logging.info(f'total amounf of rows: {new_df.count()[0]}')

train_df = new_df.sample(frac = 0.85)
logging.info(f'amounf of rows train/test set: {train_df.count()[0]}')
validation_df = new_df.drop(train_df.index)
logging.info(f'amounf of rows validation set: {validation_df.count()[0]}')

new_df.to_csv('alldata_prepped.csv', index=False)
train_df.to_csv('train_prepped.csv', index=False)
validation_df.to_csv('vali_prepped.csv', index=False)

#write_table(new_df, "prepped_20210601")