import logging

import pandas as pd

from fast_ml_linear_classification_models.services.operations.cross_validate import CrossValidationOperation
from fast_ml_linear_classification_models.services.operations.read_data import DatasetReadingOperation
from fast_ml_linear_classification_models.services.operations.split import DatasetSplittingOperation
from fast_ml_linear_classification_models.services.operations.train import TrainingOperation
from fast_ml_linear_classification_models.services.pipeline.settings import ModelSavingSettings
from fast_ml_linear_classification_models.services.utils.file_utils import get_directory, get_filepath
from fast_ml_linear_classification_models.services.utils.logging_utils import get_logger


class TrainingPipeline:
    def __init__(
            self,
            dataset_reading_op: DatasetReadingOperation,
            dataset_splitting_op: DatasetSplittingOperation,
            training_op: TrainingOperation,
            cross_validation_op: CrossValidationOperation
    ):
        self._dataset_reading_op = dataset_reading_op
        self._dataset_splitting_op = dataset_splitting_op
        self._training_op = training_op
        self._cross_validation_op = cross_validation_op

    def train(self, algorithm: str, filepath: str):
        working_directory: str = get_directory(filepath)
        logger: logging.Logger = get_logger(log_file_directory=working_directory)
        logger.info(f"[{algorithm}] Started model training")

        data: pd.DataFrame = self._dataset_reading_op.read(filepath=filepath)
        X, y = self._dataset_splitting_op.split(dataframe=data)
        model = self._training_op.train(algorithm=algorithm, X=X, y=y, directory=working_directory)
        logger.info(f"[{algorithm}] Ended model training")

        logger.info(f"[{algorithm}] Started Leave-One-Out cross validation")
        r2, mse, mae = self._cross_validation_op.run(model=model, X=X, y=y)
        logger.info(f"[{algorithm}] Ended cross validation."
                    f" R^2: {r2:.3f}, mse: {mse:.3f}, mae: {mae:.3f}")

        metrics_path: str = get_filepath(directory=working_directory, filename=ModelSavingSettings.metrics_filename)
        metrics_file = open(metrics_path, "a")
        metrics_file.write(",".join([algorithm, str(r2), str(mse), str(mae) + "\n"]))
        return f"{algorithm} model trained successfully"
