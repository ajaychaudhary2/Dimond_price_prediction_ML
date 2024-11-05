
from DimondPricePrediction.components.data_ingestion import DataIngestion
from DimondPricePrediction.components.data_transformation import DataTransformation
from DimondPricePrediction.components.model_trainer import ModelTrainer
from DimondPricePrediction.components.model_evalution import ModelEvaluation
from DimondPricePrediction.logger import logging
from DimondPricePrediction.exception import customexception
import os
import sys
import pandas as pd


obj=DataIngestion()

train_data_path,test_data_path =obj.initiate_data_ingestion()

data_transformation = DataTransformation()

train_arr , test_arr = data_transformation.initiate_data_transformation(train_data_path,test_data_path)

model_trainer = ModelTrainer()

model_trainer.initate_model_training(train_arr,test_arr)

model_evel_obj = ModelEvaluation()

model_evel_obj.initiate_model_evaluation(train_arr,test_arr)
