import os
import sys
import pandas as pd
from scipy.sparse import csr_matrix

sys.path.insert(0, '../src')
from src.exception import CustomException
from src.logger import logging
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from dataclasses import dataclass

@dataclass
class DataTransformationConfig:
    processed_data_path: str = "processed_data.csv"

class DataTransformation:
    def __init__(self):
        self.transformation_config = DataTransformationConfig()

    def initiate_data_transformation(self, train_data_path, test_data_path):
        try:
            logging.info("Started data transformation process")

            # Load the data
            train_df = pd.read_csv(train_data_path)
            test_df = pd.read_csv(test_data_path)

            # Separate features and target variable
            X_train = train_df.drop(columns=['SalePrice'])  # Adjust with your target column name
            y_train = train_df['SalePrice']
            X_test = test_df.drop(columns=['SalePrice'])  # Adjust with your target column name
            y_test = test_df['SalePrice']

            # Preprocessing pipeline
            numeric_features = X_train.select_dtypes(include=['float64', 'int64']).columns
            categorical_features = X_train.select_dtypes(include=['object']).columns

            numeric_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ])

            categorical_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(handle_unknown='ignore'))
            ])

            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', numeric_transformer, numeric_features),
                    ('cat', categorical_transformer, categorical_features)
                ])

            # Fit and transform on training data
            X_train_processed = preprocessor.fit_transform(X_train)
            X_test_processed = preprocessor.transform(X_test)

            # Convert sparse matrix to dense matrix for KNeighborsRegressor
            X_train_processed = X_train_processed.toarray() if isinstance(X_train_processed, csr_matrix) else X_train_processed
            X_test_processed = X_test_processed.toarray() if isinstance(X_test_processed, csr_matrix) else X_test_processed

            # Save processed data (optional)
            pd.DataFrame(X_train_processed).to_csv(self.transformation_config.processed_data_path, index=False)

            logging.info("Data transformation completed")

            return X_train_processed, X_test_processed, y_train, y_test

        except Exception as e:
            raise CustomException(e, sys)

if __name__ == "__main__":
    # Example usage (adjust paths as needed)
    train_data_path = os.path.join("artifacts", "train.csv")
    test_data_path = os.path.join("artifacts", "test.csv")

    transformation = DataTransformation()
    train_arr, test_arr, _, _ = transformation.initiate_data_transformation(train_data_path, test_data_path)

    print(f"Shape of transformed train data: {train_arr.shape}")
    print(f"Shape of transformed test data: {test_arr.shape}")
