import pandas as pd
import numpy as np
import sys, os, re
import math
from dotenv import load_dotenv
import box
load_dotenv(verbose=False)
sys.path.append(os.getenv("ROOT_DIR"))

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import PowerTransformer
from scipy.signal import argrelextrema
from nltk import download as nltkdownload
from nltk.corpus import stopwords
from nltk.stem.cistem import Cistem
from string import punctuation
from string import digits
from gensim import models as gensimmodels

from ja_pk.utils import basic
#from db.get_dbtable_data import get_dbtable_data

def drop_na_rows(df, special_col=""):
    if special_col == "":
        for col in df.columns:
            df = df[~pd.isnull(df[col])]
    else:
        df = df[~pd.isnull(df[special_col])]
    return df

def remove_cols(df, cols):
    if type(cols) == box.box_list.BoxList:
        df = df.drop(columns=cols)
    if type(cols) == str:
        df = df.drop(columns=[cols])
    return df
    
def mean_by_group(df, targetcol, groupcols):
    return df.groupby(groupcols)[targetcol].mean()

def new_col_substract(df, base_col, sub_col, new_col):
    df[new_col] = df[base_col] - df[sub_col]
    return df

def new_col_divide(df, base_col, divby_col, new_col):
    df[new_col] = df[base_col] / df[divby_col]
    return df

def col_past_percent_delta(df, cols, days):
    """Percentage of decrease / increase of a column compared to minus x days 

    Args:
        df (dataFrame): the dataframe
        base_col (str): the regarded numeric column
        new_col (str): name of the new column
        days (int): which day in the past shall be used for comparison

    Returns:
        [type]: [description]
    """
    if type(cols) == box.box_list.BoxList:
        for col in cols:
            new_col_name = col + "_changeperc_" + str(days)
            df[new_col_name] = (df[col] / df[col].shift(days)) * 100 - 100
    if type(cols) == str:
        new_col_name = cols + "_changeperc_" + str(days)
        df[new_col_name] = (df[cols] / df[cols].shift(days)) * 100 - 100
    return df

def shift_cols_preview(df, cols, days):
    """add the value of base col x days in the future to the current row

    Args:
        df (dataFrame): the dataframe
        cols (str): the columns to be shifted
        days (int): which day in the future shall be added to the current rows

    Returns:
        [type]: [description]
    """
    if type(cols) == box.box_list.BoxList:
        for col in cols:
            new_col_name = col + "_shiftprev_" + str(days)
            df[new_col_name] = df[col].shift(-days)
    if type(cols) == str:
        new_col_name = cols + "_shiftprev_" + str(days)
        df[new_col_name] = df[cols].shift(-days)
    return df

def rolling_mean(df, cols, days):
    if type(cols) == box.box_list.BoxList:
        for col in cols:
            new_col_name = col + "_mavg_" + str(days)
            df[new_col_name] = df[col].rolling(days).mean()
    if type(cols) == str:
        new_col_name = cols + "_mavg_" + str(days)
        df[new_col_name] = df[cols].rolling(days).mean()
    return df

def remove_by_value(df, cols, value):
    for col in cols:
        df = df[df[col] != value]
    return df

def remove_if_smaller(df, cols, value):
    if type(cols) == box.box_list.BoxList:
        for col in cols:
            df = df[df[col] > value]
    if type(cols) == str:
        df = df[df[cols] > value]
    return df

def remove_if_greater(df, cols, value):
    if type(cols) == box.box_list.BoxList:
        for col in cols:
            df = df[df[col] < value]
    if type(cols) == str:
        df = df[df[cols] < value]
    return df

def standardscaling(df, cols):
    scaler = StandardScaler().fit(df[cols])
    X_train_scaled = scaler.transform(df[cols])
    return X_train_scaled

def minmaxscaler(df, cols):
    min_max_scaler = MinMaxScaler()
    df[cols] = min_max_scaler.fit_transform(df[cols])
    return df

def skpowertransformer(df, cols):
    powertransformer = PowerTransformer()
    df[cols] = powertransformer.fit_transform(df[cols])
    return df


def oe_encode(df, cols):
    if type(cols) == box.box_list.BoxList:
        for col in cols:
            df = pd.get_dummies(df, prefix=[col], columns = [col], drop_first=False)
    if type(cols) == str:
        df = pd.get_dummies(df, prefix=[cols], columns = [cols], drop_first=False)
    return df

def zipcode_transformer(df, col):
    """Splits 5-figured zip codes into 4 new region columns."""
    def len_check(cell):
        if len(cell) != 5:
            return False
        else:
            return True
    
    def cut_zipcode(cell, digits):
        if digits == 0:
            return cell[0:]
        return cell[0:digits]

    df[col] = df[col].astype('str')
    df = df[df[col].apply(lambda x: len_check(x))]
    for i in range(0,5):
        df[f'zip_fig_{i}'] = df[col].apply(lambda x: cut_zipcode(x, i))
    return df

def less_greater_encoding(df, cols, threshold):
    """Returns 0 or 1 depending on the value in cols.

    Args:
        df (dataframe): the dataframe
        cols (str or list): the column(s) to be encoded
        threshold (str or float): median/mean of the col or float
    """
    def func(cell, thresh):
        if cell >= thresh:
            return 1
        else:
            return 0

    if type(cols) == box.box_list.BoxList:
        for col in cols:
            # if you pass e.g. mean or median as threshold:
            if type(threshold) == str:
                thresh = getattr(df[col], threshold)()
            # if you pass a numeric as threshold:
            else:
                thresh = threshold
            new_col_name = col + "_lessgreatenc"
            df[new_col_name] = df.apply(lambda x: func(x[col], thresh), axis=1)
    if type(cols) == str:
        # if you pass e.g. mean or median as threshold:
        if type(threshold) == str:
            thresh = getattr(df[cols], threshold)()
        # if you pass a numeric as threshold:
        else:
            thresh = threshold
        new_col_name = cols + "_lessgreatenc"
        df[new_col_name] = df.apply(lambda x: func(x[cols], thresh), axis=1)
    return df

def two_cols_percent_delta(df, base_col, second_col, new_col):
    df[new_col] = (df[second_col] / df[base_col]) * 100 - 100
    return df

def keep_dtype_only(df, cols, dtype):
    def type_check(x):
        return type(x) is eval(dtype)

    for col in cols:
        df = df[df[col].apply(lambda x: type_check(x))]
    return df

def set_col_dtype(df, cols, dtype):
    if type(cols) == box.box_list.BoxList:
        for col in cols:
            if dtype == 'float':
                df[col] = df[col].replace('', np.nan)
            df[col] = df[col].astype(dtype)
    if type(cols) == str:
        if dtype == 'float':
            df[col] = df[col].replace('', np.nan)
        df[cols] = df[cols].astype(dtype)
    return df

def fill_na_with_last_value(df, cols):
    """This function fills empty cells of column with last filled value.

    Args:
        df (pandas dataframe): The dataframe passed through the prep pipeline
        cols (BoxList or str): The Columns on which this function shall be applied
    """
    dfa = df.copy()
    df[cols] = dfa[cols].fillna(method='ffill')
    return df

def set_positive_values_to_zero(df, cols):
    def func(cell):
        if cell >= 0:
            return cell
        else:
            return 0

    if type(cols) == box.box_list.BoxList:
        for col in cols:
            df[col] = df.apply(lambda x: func(x[col]), axis=1)
    if type(cols) == str:
        df[cols] = df.apply(lambda x: func(x[cols]), axis=1)
    return df
    
def rename_col(df, old_colname, new_colname):
    df = df.rename(columns={old_colname: new_colname})
    return df

def lin_reg_of_col(df, X_cols, y_col, new_colname):
    # NA values are a problem here
    # Works only with date columns in X_cols
    reg = LinearRegression().fit(df[X_cols].apply(lambda x: float(x.strftime('%Y%d%m'))).values.reshape(-1, 1), df[y_col])
    df[new_colname] = reg.predict(df[X_cols].apply(lambda x: float(x.strftime('%Y%d%m'))).values.reshape(-1, 1))
    return df

def count_past_rows_with_value(df, col, value):
    row_counts = []
    nrow = df[col].count() - 1
    row_counts_index = 0
    #for element in df.iloc[nrow:0][col]:
    for element in df[col]:

        if row_counts_index > 0:
            last_row_count = row_counts[row_counts_index - 1]
            if element == value:
                row_counts.append(last_row_count + 1)
                row_counts_index += 1
            if element != value:
                row_counts.append(0)
                row_counts_index += 1

        if row_counts_index == 0:
            if element == value:
                row_counts.append(1)
                row_counts_index += 1
            if element != value:
                row_counts.append(0)
                row_counts_index += 1
    new_colname = col + "_rowcount"
    df[new_colname] = row_counts
    return df

def min_max_channel_cols(df, col, n_points):
    """Adds two columns respresenting a trend channel based on local minima and maxima.

    Args:
        df (pandas dataframe): the dataframe.
        col (float): The column for which we want to get a trend channel.
        n_points (int): amount of points in the window
    """
    col_min = f'{col}_min_{n_points}'
    col_max = f'{col}_max_{n_points}'

    df[col_min] = df.iloc[argrelextrema(df[col].values, np.less_equal,
                        order=n_points)[0]][col]
    df[col_max] = df.iloc[argrelextrema(df[col].values, np.greater_equal,
                        order=n_points)[0]][col]

    df[col_min] = df[col_min].interpolate(method='linear')
    df[col_max] = df[col_max].interpolate(method='linear')

    return df

def minmax_channel_pos(df, col, col_min, col_max):
    """Calculates percentual position in minmax channel.

    Args:
        df (pandas df): the dataframe
        col (float): the base column of the minmax channel
        col_min (float): the lower border of the minmax channel
        col_max (float): the top border of the minmax channel
    """
    df['max_min_diff'] = df[col_max] - df[col_min]
    df[col + '_chanpos'] = (df[col_max] - df[col]) / df['max_min_diff']
    return df
    
def remove_umlaute_from_colnames(df):
    clean_names = []
    for name in df.columns:
        new_name = name.replace("ü", "ue").replace("ä", "ae").replace("ö", "oe")
        clean_names.append(new_name)
    df.columns = clean_names
    return df

def remove_stopwords_umlaute_german(df, cols):
    nltkdownload('stopwords')
    german_stop_words = stopwords.words('german')
    remove_pun = str.maketrans('', '', punctuation)
    remove_digits = str.maketrans('', '', digits)

    def umlauts(text):
        """
        Replace umlauts for a given text
        
        :param word: text as string
        :return: manipulated text as str
        """
        
        tempVar = text # local variable
        
        # Using str.replace() 
        
        tempVar = tempVar.replace('ä', 'ae')
        tempVar = tempVar.replace('ö', 'oe')
        tempVar = tempVar.replace('ü', 'ue')
        tempVar = tempVar.replace('Ä', 'Ae')
        tempVar = tempVar.replace('Ö', 'Oe')
        tempVar = tempVar.replace('Ü', 'Ue')
        tempVar = tempVar.replace('ß', 'ss')
        tempVar = tempVar.replace(',', '')

        return tempVar
    
    german_stop_words_to_use = []   # List to hold words after conversion
    for word in german_stop_words:
        german_stop_words_to_use.append(umlauts(word))

    def cell_stopword_removel(cell):
        if type(cell) == float:
            return ""
        new_text = umlauts(cell)
        text_wo_pun = new_text.translate(remove_pun)
        text_wo_num = text_wo_pun.translate(remove_digits)
        text_wo_stop_words = [word for word in text_wo_num.split() if word.lower() not in german_stop_words_to_use]
        text_wo_stop_words = ' '.join(text_wo_stop_words)
        return text_wo_stop_words

    if type(cols) == box.box_list.BoxList:
        for col in cols:
            df[col] = df.apply(lambda x: cell_stopword_removel(x[col]), axis=1)
    if type(cols) == str:
        df[cols] = df.apply(lambda x: cell_stopword_removel(x[cols]), axis=1)

    return df

def stemming_german(df, cols):

    cs = Cistem()

    def cell_stem(cell):
        if type(cell) == float:
            return ""
        text_stemmed_words = [cs.stem(word) for word in cell.split()]
        text_stemmed_words = ' '.join(text_stemmed_words)
        return text_stemmed_words

    if type(cols) == box.box_list.BoxList:
        for col in cols:
            df[col] = df.apply(lambda x: cell_stem(x[col]), axis=1)
    if type(cols) == str:
        df[cols] = df.apply(lambda x: cell_stem(x[cols]), axis=1)

    return df

def word2vectorizer(df, cols):
    df_wv = df[cols]
    df_wv_onecol = df_wv.apply(lambda x: ','.join(x.astype(str)), axis=1)
    df_clean = pd.DataFrame({'clean': df_wv_onecol})

    word_lists = [row.split(',') for row in df_clean['clean']]
    model = gensimmodels.Word2Vec(word_lists)

    # model.train(word_lists, total_examples = 100000, epochs = 200 - train the model
    # len(model.wv) - for how many keys do we have vectors?
    # model.wv[4] - get one vecotr
    # model.wv['mein key'] - get vector to key, if it exists
    print('fertig')