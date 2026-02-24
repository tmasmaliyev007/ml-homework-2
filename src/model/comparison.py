from sklearn.linear_model import LinearRegression, PoissonRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error


class RegressionComparison:
    """Linear and Poisson regression on a count-like target."""

    @staticmethod
    def fit_linear(X_train, X_test, y_train, y_test) -> dict:
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        return {
            "model": model,
            "y_pred": y_pred,
            "mse": mean_squared_error(y_test, y_pred),
            "mae": mean_absolute_error(y_test, y_pred),
            "r2": model.score(X_test, y_test),
        }

    @staticmethod
    def fit_poisson(X_train, X_test, y_train, y_test) -> dict:
        model = PoissonRegressor(alpha=0.01, max_iter=1000)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        return {
            "model": model,
            "y_pred": y_pred,
            "mse": mean_squared_error(y_test, y_pred),
            "mae": mean_absolute_error(y_test, y_pred),
            "d2": model.score(X_test, y_test),
        }
