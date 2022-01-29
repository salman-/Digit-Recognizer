import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer


class Dataset:

    def __init__(self, dt_path):
        dt = pd.read_csv(dt_path)
        pixel_columns = dt.columns[1:]

        column_transformer = ColumnTransformer(transformers=[("min_max_scaler", MinMaxScaler(), pixel_columns)])
        self.dt = column_transformer.fit_transform(dt.iloc[:, 1:])

    def get_test_train_datasets(self):
        dt = pd.read_csv("../datasets/train.csv")
        train_x, test_x, train_y, test_y = train_test_split(self.dt, dt['label'], train_size=.8, shuffle=True)
        return train_x, test_x, train_y, test_y
