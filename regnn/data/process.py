import pandas as pd
import numpy as np
from typing import Union, Sequence


# preprocess categorical col to one hot encoding
def binary_to_one_hot(df, binary_cats: Sequence, dtype="float"):
    one_hot_maps = {}
    if binary_cats is None:
        binary_cats = df.columns
    for bc in binary_cats:
        df[bc] = df[bc].cat.codes
    return df, binary_cats


def multi_cat_to_one_hot(df, multi_cats: Sequence, dtype="float"):
    if multi_cats is None:
        multi_cats = df.columns
    df2 = pd.get_dummies(
        df[multi_cats], columns=multi_cats, dtype=float, drop_first=True
    )
    new_colnames = df2.columns
    if dtype == "category":
        for c in df2.columns:
            df2[c] = df2[c].astype(dtype)
    df = pd.concat([df, df2], axis=1)
    df.drop(multi_cats, inplace=True, axis=1)
    return df, new_colnames


def standardize_cols(df, columns: Sequence):
    if columns is None:
        columns = df.columns
    mean_std_dict = {}
    for c in columns:
        if df[c].dtype != "category":
            mean = df[c].mean()
            std = df[c].std()
            df[c] = (df[c] - mean) / std
            mean_std_dict[c] = (mean, std)
        else:
            print("is category: ", c)
    return df, mean_std_dict


# convert ordinal columns to integer values
def convert_categorical_to_ordinal(df, ordinal_cols: Sequence):
    if ordinal_cols is None:
        ordinal_cols = df.columns
    for c in ordinal_cols:
        df[c] = df[c].cat.codes
        df[c] = df[c].astype("float")
    return df, ordinal_cols


def map_to_zero_one(df, cols: Sequence):
    for c in cols:
        col_min = df[c].min()
        col_max = df[c].max()
        df[c] = (df[c] - col_min) / (col_max - col_min)
    return df, cols
