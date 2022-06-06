from fast_ml_linear_classification_models.proto.compiled import linear_classification_pb2_grpc
from fast_ml_linear_classification_models.proto.compiled.linear_classification_pb2 import TrainingResponse, PredictionResponse
from fast_ml_linear_classification_models.services.operations.cross_validate import CrossValidationOperation
from fast_ml_linear_classification_models.services.operations.load_model import ModelLoadingOperation
from fast_ml_linear_classification_models.services.operations.predict import PredictionOperation
from fast_ml_linear_classification_models.services.operations.read_data import DatasetReadingOperation
from fast_ml_linear_classification_models.services.operations.split import DatasetSplittingOperation
from fast_ml_linear_classification_models.services.operations.train import TrainingOperation
from fast_ml_linear_classification_models.services.pipeline.predict import PredictionPipeline
from fast_ml_linear_classification_models.services.pipeline.train import TrainingPipeline


class LinearRegressionService(linear_classification_pb2_grpc.LinearClassificationService):
    def Train(self, request, context, **kwargs):
        dataset_reading_op: DatasetReadingOperation = DatasetReadingOperation()
        dataset_splitting_op: DatasetSplittingOperation = DatasetSplittingOperation()
        training_op: TrainingOperation = TrainingOperation()
        cross_validation_op: CrossValidationOperation = CrossValidationOperation()

        pipeline: TrainingPipeline = TrainingPipeline(
            dataset_reading_op=dataset_reading_op,
            dataset_splitting_op=dataset_splitting_op,
            training_op=training_op,
            cross_validation_op=cross_validation_op
        )

        return TrainingResponse(
            configuration=pipeline.train(algorithm=request.algorithm, filepath=request.filepath)
        )

    def Predict(self, request, context, **kwargs):
        dataset_reading_op: DatasetReadingOperation = DatasetReadingOperation()
        model_loading_op: ModelLoadingOperation = ModelLoadingOperation()
        prediction_op: PredictionOperation = PredictionOperation()

        pipeline: PredictionPipeline = PredictionPipeline(
            dataset_reading_op=dataset_reading_op,
            model_loading_op=model_loading_op,
            prediction_op=prediction_op
        )

        return PredictionResponse(
            message=pipeline.predict(algorithm=request.algorithm, filepath=request.filepath)
        )
