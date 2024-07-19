import os
import sys
import numpy as np 
import pandas as pd
import dill
import pickle
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException
from src.logger import logging

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
    except Exception as e:
        raise CustomException(e, sys)
    
def evaluate_models(X_train, y_train, X_test, y_test, models, param):
    try:
        report = {}

        for i in range(len(list(models))):
            model_name = list(models.keys())[i]
            model = list(models.values())[i]
            para = param[model_name]

            logging.info(f"Evaluating model: {model_name}")
            
            # Convert sparse matrix to dense matrix for KNeighborsRegressor
            if model.__class__.__name__ == 'KNeighborsRegressor':
                logging.info("Converting sparse matrix to dense for KNeighborsRegressor")
                X_train = X_train.toarray() if hasattr(X_train, 'toarray') else X_train
                X_test = X_test.toarray() if hasattr(X_test, 'toarray') else X_test

            logging.info(f"Performing GridSearchCV for model: {model_name}")
            gs = GridSearchCV(model, para, cv=3)
            gs.fit(X_train, y_train)

            best_params = gs.best_params_
            logging.info(f"Best parameters for {model_name}: {best_params}")
            
            model.set_params(**best_params)
            model.fit(X_train, y_train)

            logging.info(f"Predicting on training data for model: {model_name}")
            y_train_pred = model.predict(X_train)
            
            logging.info(f"Predicting on test data for model: {model_name}")
            y_test_pred = model.predict(X_test)

            logging.info(f"Calculating R2 score for model: {model_name}")
            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)

            report[model_name] = test_model_score
            logging.info(f"Model {model_name} evaluation completed with test score: {test_model_score}")

        return report
    except KeyError as ke:
        logging.error(f"KeyError occurred during model evaluation: {ke}")
        raise CustomException(ke, sys)
    except Exception as e:
        logging.error(f"An error occurred during model evaluation: {e}")
        raise CustomException(e, sys)
    
def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        raise CustomException(e, sys)
