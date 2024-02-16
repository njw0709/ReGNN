import pandas as pd
import numpy as np


# preprocess categorical col to one hot encoding
def binary_to_one_hot(df, binary_cats, dtype=float):
    one_hot_maps = {}
    for bc in binary_cats:
        unique_vals = df[bc].unique()
        if isinstance(unique_vals[0], str):
            one_hot_map = {c: np.float32(c[0]) for c in unique_vals}
            df[bc] = df[bc].map(one_hot_map).astype(dtype)
    return df


def multi_cat_to_one_hot(df, multi_cats, dtype=float):
    df2 = pd.get_dummies(df[multi_cats], columns=multi_cats, dtype=float)
    if dtype == "category":
        for c in df2.columns:
            df2[c] = df2[c].astype(dtype)
    df = pd.concat([df, df2], axis=1)
    df.drop(multi_cats, inplace=True, axis=1)
    return df


def standardize_continuous_cols(df, continuous_vars):
    mean_std_dict = {}
    for c in continuous_vars:
        mean = df[c].mean()
        std = df[c].std()
        df[c] = (df[c] - mean) / std
        mean_std_dict[c] = (mean, std)
    return df, mean_std_dict


# convert ordinal columns to integer values
def convert_categorical_to_ordinal(df, ordinal_cols):
    for c in ordinal_cols:
        unique_values = df[c].unique()
        if isinstance(unique_values[0], str):
            vals = {val: int(val[0]) for val in unique_values}

        else:
            vals = {val: int(val) for val in unique_values}
        for v in unique_values:
            df[c] = df[c].replace(v, vals[v])
        df[c] = df[c].astype("int32")
    return df
