import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import numpy as np
from sklearn.decomposition import PCA

class Preprocessor:
    def __init__(self, data, config):
        self.data = data.copy()
        self.processed_data = None
        self.unwanted_columns = config['unwanted_columns']
        self.categorical_columns = config['categorical_columns']

    def process(self):
        self._drop_unwanted_columns()
        self._one_hot_encode()
        self._handle_inf_values()
        self._handle_nan_values()
        self.processed_data = self.data  # 更新されたデータを保存

    def _drop_unwanted_columns(self):
        self.data.drop(columns=self.unwanted_columns, inplace=True)

    def _one_hot_encode(self):
        encoder = OneHotEncoder(sparse_output=False, categories='auto')
        encoded_data = encoder.fit_transform(self.data[self.categorical_columns])
        encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(self.categorical_columns), index=self.data.index)
        self.data = pd.concat([self.data.drop(columns=self.categorical_columns), encoded_df], axis=1)


    def _handle_inf_values(self):
        self.data.replace([np.inf, -np.inf], np.nan, inplace=True)

    def _handle_nan_values(self):
        self.data.fillna(self.data.median(), inplace=True)

    def get_processed_data(self):
        if self.processed_data is None:
            raise Warning("You should call the 'process()' method before getting the processed data.")
        return self.processed_data
