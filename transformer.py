import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder

class Transformer:
    def __init__(self, data, config):
        self.data = data.copy()
        self.unwanted_columns = config['unwanted_columns']
        self.categorical_columns = config['categorical_columns']

    def transform(self):
        self._drop_unwanted_columns()
        self._one_hot_encode()
        self._handle_inf_values() # one-hotの後に実行
        self._handle_nan_values()
        return self.data

    def _drop_unwanted_columns(self):
        self.data.drop(columns=self.unwanted_columns, inplace=True)
    
    def _handle_inf_values(self):
        self.data.replace([np.inf, -np.inf], np.nan, inplace=True)

    def _handle_nan_values(self):
        self.data.fillna(self.data.median(), inplace=True)

    def _one_hot_encode(self): # 訓練データとテストデータを同時にone-hotエンコードするのは間違っているので後で修正
        encoder = OneHotEncoder(sparse_output=False, categories='auto')
        encoded_data = encoder.fit_transform(self.data[self.categorical_columns])
        encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(self.categorical_columns), index=self.data.index)
        self.data = pd.concat([self.data.drop(columns=self.categorical_columns), encoded_df], axis=1)
