import pandas as pd
import numpy as np
import os
import sys
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from DimondPricePrediction.logger import logging
from DimondPricePrediction.exception import customexception
from DimondPricePrediction.utils.utils import save_object

class DataTransformation_config:
    preprocessor_obj_file_path = os.path.join("artifacts", "preprocessor.pkl")

class DataTransformation:
    
    def __init__(self):
        self.data_transformation_config = DataTransformation_config()
    
    def get_data_transformation(self):
        try:
            logging.info("Data transformation initiated")

            # Define categorical and numerical columns
            categorical_cols = ['cut', 'color', 'clarity']
            numerical_cols = ['carat', 'depth', 'table', 'x', 'y', 'z']

            # Define the custom ranking for each ordinal variable
            cut_categories = ["Fair", "Good", "Very Good", "Premium", "Ideal"]
            color_categories = ["D", "E", "F", "G", "H", "I", "J"]
            clarity_categories = ["I1", "SI2", "SI1", "VS2", "VS1", "VVS2", "VVS1", "IF"]

            logging.info("Pipeline initiated")

            # Numerical pipeline
            num_pipeline = Pipeline(
                steps=[
                    ("Imputer", SimpleImputer(strategy="mean")),
                    ("Scaler", StandardScaler())
                ]
            )

            # Categorical pipeline
            cat_pipeline = Pipeline(
                steps=[
                    ("Imputer", SimpleImputer(strategy="most_frequent")),
                    ("Ordinal Encoding", OrdinalEncoder(categories=[cut_categories, color_categories, clarity_categories]))
                ]
            )

            # Preprocessor
            preprocessor = ColumnTransformer(
                transformers=[
                    ("num_pipeline", num_pipeline, numerical_cols),
                    ("cat_pipeline", cat_pipeline, categorical_cols)
                ]
            )

            return preprocessor

        except Exception as e:
            logging.info("Exception occurred during data transformation stage")
            raise customexception(e, sys)

    def initiate_data_transformation(self, train_data_path, test_data_path):
        try:
            train_df = pd.read_csv(train_data_path)
            test_df = pd.read_csv(test_data_path)

            logging.info("Read train and test data completed")
            logging.info(f"Train DataFrame head:\n{train_df.head()}")
            logging.info(f"Test DataFrame head:\n{test_df.head()}")
            logging.info(f"Data types:\n{train_df.dtypes}")

            # Ensure numeric columns, especially 'z' which may have non-numeric values
            train_df['z'] = pd.to_numeric(train_df['z'], errors='coerce')
            test_df['z'] = pd.to_numeric(test_df['z'], errors='coerce')

            # Get the preprocessing object
            preprocessor_obj = self.get_data_transformation()

            # Define target column and drop columns
            target_column_name = 'price'
            drop_columns = [target_column_name, "id"]

            # Separate input features and target
            input_feature_train_df = train_df.drop(columns=drop_columns, axis=1)
            input_feature_test_df = test_df.drop(columns=drop_columns, axis=1)

            target_feature_train_df = train_df[target_column_name]
            target_feature_test_df = test_df[target_column_name]

            # Transform the data
            input_feature_train_arr = preprocessor_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessor_obj.transform(input_feature_test_df)

            # Combine the transformed features with the target column
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info("Applying preprocessor on train and test datasets")

            # Save the preprocessor object
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessor_obj
            )

            logging.info("Preprocessor object saved successfully")

            return train_arr, test_arr

        except Exception as e:
            logging.info("Exception occurred during data transformation stage")
            raise customexception(e, sys)
