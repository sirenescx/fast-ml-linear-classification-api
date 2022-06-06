import logging

import pandas as pd

from fast_ml_linear_classification_models.services.operations.load_model import ModelLoadingOperation
from fast_ml_linear_classification_models.services.operations.predict import PredictionOperation
from fast_ml_linear_classification_models.services.operations.read_data import DatasetReadingOperation
from fast_ml_linear_classification_models.services.utils.file_utils import get_directory
from fast_ml_linear_classification_models.services.utils.logging_utils import get_logger


class PredictionPipeline:
    def __init__(
            self,
            dataset_reading_op: DatasetReadingOperation,
            model_loading_op: ModelLoadingOperation,
            prediction_op: PredictionOperation
    ):
        self._dataset_reading_op = dataset_reading_op
        self._model_loading_op = model_loading_op
        self._prediction_op = prediction_op

    def predict(self, filepath: str, algorithm: str):
        working_directory: str = get_directory(filepath)
        logger: logging.Logger = get_logger(log_file_directory=working_directory)
        logger.info(f"Started prediction with {algorithm} model")
        features: pd.DataFrame = self._dataset_reading_op.read(filepath=filepath)
        model = self._model_loading_op.load(model_config_directory=working_directory, algorithm=algorithm)
        predictions_path = self._prediction_op.predict(
            X=features,
            model=model,
            output_directory=working_directory,
            algorithm=algorithm
        )
        logger.info(f"Ended prediction with {algorithm} model")
        return f"{algorithm}"
