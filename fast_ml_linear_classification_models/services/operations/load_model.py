import pickle

from fast_ml_linear_classification_models.services.utils.file_utils import get_filepath


class ModelLoadingOperation:
    def load(self, model_config_directory: str, algorithm: str):
        model_path: str = get_filepath(
            directory=model_config_directory,
            filename=f"{algorithm}.pickle"
        )
        loaded_model = pickle.load(open(model_path, "rb"))
        return loaded_model
