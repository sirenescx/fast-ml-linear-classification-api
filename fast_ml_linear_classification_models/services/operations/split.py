import pandas as pd

from fast_ml_linear_classification_models.services.pipeline.settings import DataFrameSettings


class DatasetSplittingOperation:
    def split(self, dataframe: pd.DataFrame) -> (pd.DataFrame, pd.Series):
        features = dataframe.drop(DataFrameSettings.target_column_name, axis=1)
        target = dataframe[DataFrameSettings.target_column_name]
        return features, target
