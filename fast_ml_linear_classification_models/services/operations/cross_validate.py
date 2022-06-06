import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import LeaveOneOut


class CrossValidationOperation:
    def run(self, model: LinearRegression, X: pd.DataFrame, y: pd.Series):
        r2 = self._cross_validate(model, X, y, r2_score)
        mse = self._cross_validate(model, X, y, mean_squared_error)
        mae = self._cross_validate(model, X, y, mean_absolute_error)
        return r2, mse, mae

    def _cross_validate(self, model: LinearRegression, X: pd.DataFrame, y: pd.Series, metric):
        leave_one_out: LeaveOneOut = LeaveOneOut()
        result = np.zeros(y.shape)

        for train_index, test_index in leave_one_out.split(X):
            X_train, X_test = X.values[train_index], X.values[test_index]
            y_train, y_test = y.values[train_index], y.values[test_index]
            model.fit(X_train, y_train)
            result[test_index] = model.predict(X_test)
        mean_metric_score = metric(y, result)
        return mean_metric_score
