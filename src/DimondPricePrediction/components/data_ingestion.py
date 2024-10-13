import pandas as pd
import numpy as np
import os
import sys


from DimondPricePrediction.logger import logging
from DimondPricePrediction.exception import customexception

from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from pathlib import Path



class DataIngestion_config:
    rawdata_path:str = os.path.join("artifacts","raw.csv")
    traindata_path:str=os.path.join("artifacts","train.csv")
    testdata_path:str=os.path.join("artifacts","test.csv")




class DataIngestion:
    
    def __init__(self):
        self.DataIngestion_config=DataIngestion_config
    
    
    
    def initiate_data_ingestion(self):
        logging.info("data ingestion started")
        try:
            data=pd.read_csv(Path(os.path.join("notebooks/data","GemStone.csv")))
            logging.info("I have read the data from data set")
            
            os.makedirs(os.path.dirname(os.path.join(self.DataIngestion_config.rawdata_path)),exist_ok=True)
            data.to_csv(self.DataIngestion_config.rawdata_path,index=False)
            logging.info("succesfully save the raw data in artifacts folder")
            
            
            
        
            
            
            
            logging.info("here I am perfom  train test split")
            train_data,test_data=train_test_split(data,test_size=.25)
            logging.info("train test split   completed")
            
            data.to_csv(self.DataIngestion_config.traindata_path,index=False)

            data.to_csv(self.DataIngestion_config.testdata_path,index=False)
            logging.info("succesfully save the  train and test data in afrifact folder")
            
            
            logging.info("data ingestion part completed")            
            
            
            return(
                
                self.DataIngestion_config.traindata_path,
                self.DataIngestion_config.testdata_path
                
            )
        
            
            
        except Exception as e:
            logging.info("Exception occur durin data ingestion  stage")
            raise customexception(e,sys)