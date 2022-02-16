#!/usr/bin/env python
# -*- coding: utf-8 -*-

import glob
import csv
import string
import re
import pandas as pd
import os

import sys, os
from dotenv import load_dotenv
load_dotenv(verbose=False)
sys.path.append(os.getenv("ROOT_DIR"))
#! from db.config import Config
#! from db.connect_db import DbConnection
#! from db.execute_query import QueryExecution

def get_folder_files(foldername, pattern = '*'):
    """returns a list of the names of the files in the given folder.

    Arguments:
        foldername {str} -- name of the folder which should be browsed for files

    Keyword Arguments:
        pattern {str} -- regex pattern which is used for finding certain files (default: {'*'})

    Returns:
        [list] -- list of names of the files in the given folder
    """
    return glob.glob(foldername + '/**/' + pattern, recursive=True)

def get_parent_folder(path):
    """Returns the superior folder / parent folder of a given folder

    Args:
        path (str): the path of the child folder

    Returns:
        str: path of the parent folder / superior folder
    """
    if "\\" in path:
        path_components = path.split("\\")
    if "/" in path:
        path_components = path.split("/")
    return path_components[-2]

def get_file_headers(path):
    """Returns the headers of a csv file for naming db columns.

    It also removes special characters which should not be used as db col names.
    If there is only an empty string left as header of a certain column,
    the function returns 'leer' as header name for that column.

    Args:
        path (str): path to the csv file

    Returns:
        list: list of the headers of the csv file
    """
    with open(path, 'r') as f:
        d_reader = csv.DictReader(f)
        header = d_reader.fieldnames
        header_decoded = []
        for col in header:
            col_only_letters = re.sub(r'[^a-zA-Z]',r'', col.replace("\W", ""))
            if not col_only_letters:
                col_only_letters = 'leer'
            header_decoded.append(col_only_letters)
            
    return header_decoded

def read_query_file(path):
    """Just returns the content of a txt file.
    
    Other file formats might work, too. Not tested yet.

    Args:
        path (str): path to a txt file

    Returns:
        str: string with all the content of the file
    """
    f = open(path,"r")
    return f.read()


def delete_na_from_csv(file_path):
    """Delete rows with missing values from a csv file.

    Args:
        file_path (str): path to csv file
    """
    df = pd.read_csv(file_path, sep=',').dropna()
    df.to_csv(file_path, index=False)

def value_sample_pd_table(pd_df):
    """Return the first row of a dataframe.

    Args:
        pd_df (df): pandas dataframe

    Returns:
        df: pandas dataframe (one row)
    """
    col_list = list(pd_df.columns)   # - with colnames
    # result = [pd_df.loc[[0], [col]] for col in col_list] - first cell, but including colname
    result = [pd_df.loc[0, col] for col in col_list]   # first cell value
    return result

def cols_pd_table(pd_df):
    """Returns column names of a pandas dataframe as a list

    Args:
        pd_df (dataframe): Pandas dataframe

    Returns:
        list: list of column names
    """
    return list(pd_df.columns)

def read_config(path):
    """Reads config file and returns the content as list-kinda object

    Afterwards, you can retrieve content out of the config object by the syntax cfg_obj.param_area.param

    Args:
        path (str): path to config file

    Returns:
        list: config list object
    """
    # Read exp config yml
    import yaml
    from box import Box
    with open(os.getenv("ROOT_DIR") + path, "r") as ymlfile:
        exp_config = Box(yaml.safe_load(ymlfile))
    return exp_config

#! def setup_db_connection():
#!    dbini_path = os.getenv("ROOT_DIR")
#!    my_DbConnection = DbConnection(dbini_path)
#!    connection_objects = my_DbConnection.setup_connection()
#!    conn = connection_objects[0]
#!    cur = connection_objects[1]
#!
#!    # return connection and cursor
#!    return conn, cur