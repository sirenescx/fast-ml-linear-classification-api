import pickle
import sys
from logging import Logger

import pandas as pd
from sklearn.linear_model import (
    Ridge, Lasso, ElasticNet, SGDRegressor, LinearRegression, Lars,
    HuberRegressor, RANSACRegressor, LassoLars, OrthogonalMatchingPursuit,
    TheilSenRegressor, QuantileRegressor
)

from fast_ml_linear_classification_models.services.utils.file_utils import get_filepath

from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer

from fast_ml_linear_classification_models.services.utils.logging_utils import get_logger


class TrainingOperation:
    def train(self, algorithm: str, X: pd.DataFrame, y: pd.Series, directory: str):
        logger: Logger = get_logger(log_file_directory=directory)

        if algorithm == "LinearRegression":
            model = self._train_linear_regression(X, y)
            logger.info(f"[{algorithm}] {model.get_params()}")
            self._save_model(model=model, directory=directory, algorithm=algorithm)
            return model

        if algorithm == "Lars":
            model = self._train_lars(X, y)
            logger.info(f"[{algorithm}] {model.get_params()}")
            self._save_model(model=model, directory=directory, algorithm=algorithm)
            return model

        logger.info(f"[{algorithm}] Started optimizing hyperparameters")

        match algorithm:
            case "Ridge":
                model = self._train_ridge(X, y)
            case "Lasso":
                model = self._train_lasso(X, y)
            case "LassoLars":
                model = self._train_lasso_lars(X, y)
            case "OrthogonalMatchingPursuit":
                model = self._train_orthogonal_matching_pursuit(X, y)
            case "ElasticNet":
                model = self._train_elastic_net(X, y)
            case "SGDRegressor":
                model = self._train_sgd_regressor(X, y)
            case "HuberRegressor":
                model = self._train_huber_regressor(X, y)
            case "RANSACRegressor":
                model = self._train_ransac_regressor(X, y)
            case "TheilSenRegressor":
                model = self._train_theil_sen_regressor(X, y)
            case "QuantileRegressor":
                model = self._train_quantile_regressor(X, y)
            case "ARDRegression":
                model = self._train_ard_regression(X, y)
            case _:
                raise Exception("Invalid algorithm name")

        logger.info(f"[{algorithm}] Ended optimizing hyperparameters")
        logger.info(f"[{algorithm}] {model.get_params()}")

        self._save_model(model=model, directory=directory, algorithm=algorithm)

        return model

    def _save_model(self, model, directory: str, algorithm: str):
        model_path: str = get_filepath(directory=directory, filename=f"{algorithm}.pickle")
        pickle.dump(model, open(model_path, "wb"))

    def _train_linear_regression(self, X: pd.DataFrame, y: pd.Series):
        model = LinearRegression()
        model.fit(X, y)
        return model

    def _train_ridge(self, X: pd.DataFrame, y: pd.Series):
        optimizer = BayesSearchCV(
            estimator=Ridge(),
            search_spaces={
                "alpha": Real(1e-6, 1e+6, prior="log-uniform")
            },
            n_iter=50,
            random_state=42
        )
        _ = optimizer.fit(X, y)
        return optimizer.best_estimator_

    def _train_lasso(self, X: pd.DataFrame, y: pd.Series):
        optimizer = BayesSearchCV(
            estimator=Lasso(),
            search_spaces={
                "alpha": Real(1e-6, 1e+6, prior="log-uniform")
            },
            n_iter=50,
            random_state=42
        )
        _ = optimizer.fit(X, y)
        return optimizer.best_estimator_

    def _train_elastic_net(self, X: pd.DataFrame, y: pd.Series):
        optimizer = BayesSearchCV(
            estimator=ElasticNet(),
            search_spaces={
                "alpha": Real(1e-6, 1e+6, prior="log-uniform"),
                "l1_ratio": Real(sys.float_info.epsilon, 1, prior="log-uniform")
            },
            n_iter=50,
            random_state=42
        )
        _ = optimizer.fit(X, y)
        return optimizer.best_estimator_

    def _train_lars(self, X: pd.DataFrame, y: pd.Series):
        model = Lars()
        model.fit(X, y)
        return model

    def _train_lasso_lars(self, X: pd.DataFrame, y: pd.Series):
        optimizer = BayesSearchCV(
            estimator=LassoLars(),
            search_spaces={
                "alpha": Real(1e-6, 1e+6, prior="log-uniform")
            },
            n_iter=50,
            random_state=42
        )
        _ = optimizer.fit(X, y)
        return optimizer.best_estimator_

    def _train_orthogonal_matching_pursuit(self, X: pd.DataFrame, y: pd.Series):
        optimizer = BayesSearchCV(
            estimator=OrthogonalMatchingPursuit(),
            search_spaces={
                "n_nonzero_coefs": Integer(1, X.shape[1] - 1),
            },
            n_iter=50,
            random_state=42
        )
        _ = optimizer.fit(X, y)
        return optimizer.best_estimator_

    def _train_bayesian_regression(self, X: pd.DataFrame, y: pd.Series):
        pass

    def _train_sgd_regressor(self, X: pd.DataFrame, y: pd.Series):
        optimizer = BayesSearchCV(
            estimator=SGDRegressor(),
            search_spaces={
                "loss": Categorical(["squared_error", "huber", "epsilon_insensitive", "squared_epsilon_insensitive"]),
                "alpha": Real(1e-6, 1e+6, prior="log-uniform"),
                "penalty": Categorical(["l2", "l1", "elasticnet"]),
                "l1_ratio": Real(sys.float_info.epsilon, 1, prior="log-uniform")
            },
            n_iter=50,
            random_state=42
        )
        _ = optimizer.fit(X, y)
        return optimizer.best_estimator_

    def _train_passive_aggressive_regressor(self, X: pd.DataFrame, y: pd.Series):
        pass

    def _train_ransac_regressor(self, X: pd.DataFrame, y: pd.Series):
        optimizer = BayesSearchCV(
            estimator=RANSACRegressor(),
            search_spaces={
                "min_samples": Integer(1, X.shape[1] - 1),
                "residual_threshold": Real(1e-6, 1e+6, prior="log-uniform"),
                "max_trials": Integer(1, 1e+6, prior="log-uniform")
            },
            n_iter=50,
            random_state=42
        )
        _ = optimizer.fit(X, y)
        return optimizer.best_estimator_

    def _train_huber_regressor(self, X: pd.DataFrame, y: pd.Series):
        optimizer = BayesSearchCV(
            estimator=HuberRegressor(),
            search_spaces={
                "alpha": Real(1e-6, 1e+6, prior="log-uniform")
            },
            n_iter=50,
            random_state=42
        )
        _ = optimizer.fit(X, y)
        return optimizer.best_estimator_

    def _train_quantile_regressor(self, X: pd.DataFrame, y: pd.Series):
        optimizer = BayesSearchCV(
            estimator=QuantileRegressor(),
            search_spaces={
                "quantile": Real(0, 1, prior="log-uniform"),
                "alpha": Real(1e-6, 1e+6, prior="log-uniform"),
                "solver": Categorical(["highs-ds", "highs-ipm", "highs", "interior-point", "revised simplex"])
            },
            n_iter=50,
            random_state=42
        )
        _ = optimizer.fit(X, y)
        return optimizer.best_estimator_

    def _train_theil_sen_regressor(self, X: pd.DataFrame, y: pd.Series):
        num_features = X.shape[1] + 1
        num_samples = X.shape[0]
        if num_samples > num_features:
            search_spaces = {
                    "max_subpopulation": Integer(1, 1e+6, prior="log-uniform"),
                    "n_subsamples": Integer(num_features, num_samples, prior="log-uniform"),
                    "max_iter": Integer(10, 1000, prior="log-uniform"),
                    "tol": Real(1e-6, 1e+6, prior="log-uniform")
                }
        else:
            search_spaces = {
                "max_subpopulation": Integer(1, 1e+6, prior="log-uniform"),
                "max_iter": Integer(10, 1000, prior="log-uniform"),
                "tol": Real(1e-6, 1e+6, prior="log-uniform")
            }
        optimizer = BayesSearchCV(
            estimator=TheilSenRegressor(),
            search_spaces=search_spaces,
            n_iter=50,
            random_state=42
        )
        _ = optimizer.fit(X, y)
        return optimizer.best_estimator_

    def _train_ard_regression(self, X: pd.DataFrame, y: pd.Series):
        pass
