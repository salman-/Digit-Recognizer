import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer

class Dataset:

    def __init__(self, dt_path):
        dt = pd.read_csv(dt_path)
        self.pixel_columns = dt.columns[dt.columns != 'label']
        print(dt_path, "  ", self.pixel_columns)

        column_transformer = ColumnTransformer(transformers=[("min_max_scaler", MinMaxScaler(), self.pixel_columns)])
        self.dt = column_transformer.fit_transform(dt)

    def get_test_train_datasets(self):
        target = self.one_hot_target_feature()
        train_x, test_x, train_y, test_y = train_test_split(self.dt,
                                                            target,
                                                            train_size=.8,
                                                            shuffle=True)
        return train_x.reshape(33600, 28, 28), test_x.reshape(8400, 28, 28), train_y, test_y

    def one_hot_target_feature(self):
        dt = pd.read_csv('../datasets/train.csv')
        column_transformer = ColumnTransformer(transformers=[("one_hot_encode", OneHotEncoder(), ['label'])])
        return column_transformer.fit_transform(dt).todense()
