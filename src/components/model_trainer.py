import sys
from dataclasses import dataclass
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.metrics import r2_score

sys.path.insert(0, '../src')
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models

import logging
from src.logger import setup_logging

setup_logging()

logging.info("Testing logging setup in model_trainer.py")


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = "artifacts/model.pkl"

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting training and test input data")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )

            models = {
                "Linear Regression": LinearRegression(),
                "Lasso": Lasso(alpha=10.0),
                "Ridge": Ridge(alpha=10.0),
                "K-Neighbors Regressor": KNeighborsRegressor(),
                "Decision Tree": DecisionTreeRegressor(criterion='squared_error'),
                "Random Forest Regressor": RandomForestRegressor(n_estimators=100, criterion='squared_error'),
                "XGBRegressor": XGBRegressor(learning_rate=0.05, n_estimators=128),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False, depth=10, iterations=100, learning_rate=0.1),
                "AdaBoost Regressor": AdaBoostRegressor(learning_rate=0.1, n_estimators=256)
            }

            params = {
                "Decision Tree": {
                    'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                },
                "Random Forest Regressor": {
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                },
                "Linear Regression": {},
                "Lasso": {
                    'alpha': [0.001, 0.01, 0.1, 1.0, 10.0]
                },
                "Ridge": {
                    'alpha': [0.001, 0.01, 0.1, 1.0, 10.0]
                },
                "K-Neighbors Regressor": {
                    'n_neighbors': [5, 10, 15, 20],
                    'weights': ['uniform', 'distance'],
                    'p': [1, 2]
                },
                "XGBRegressor": {
                    'learning_rate': [0.1, 0.01, 0.05, 0.001],
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                },
                "CatBoosting Regressor": {
                    'depth': [6, 8, 10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100]
                },
                "AdaBoost Regressor": {
                    'learning_rate': [0.1, 0.01, 0.5, 0.001],
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                }
            }

            logging.info("Evaluating models")
            model_report = evaluate_models(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
                                           models=models, param=params)

            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            logging.info(f"Best model found: {best_model_name} with R2 score: {best_model_score}")

            if best_model_score < 0.6:
                logging.error("No best model found with a score above 0.6")
                raise CustomException("No best model found")

            save_object(file_path=self.model_trainer_config.trained_model_file_path, obj=best_model)

            logging.info(f"Model saved to {self.model_trainer_config.trained_model_file_path}")

            predicted = best_model.predict(X_test)
            r2_square = r2_score(y_test, predicted)
            logging.info(f"R2 score on test data: {r2_square}")

            return r2_square

        except Exception as e:
            logging.error(f"An error occurred: {e}")
            raise CustomException(e, sys)
