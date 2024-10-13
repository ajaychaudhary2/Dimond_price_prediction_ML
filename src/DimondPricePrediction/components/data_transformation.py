import pandas as pd
import numpy as np
import os
import sys

from DimondPricePrediction.logger import logging
from DimondPricePrediction.exception import customexception

from sklearn.impute import SimpleImputer 
from sklearn.preprocessing import StandardScaler,OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from DimondPricePrediction.utils.utils import save_object





class DataTransformation_config:
    preprocessor_obj_file_path=os.path.join("artifacts","preprocessor.pk1")





class DataTransformation:
    
    def __init__(self):
        self.data_transformation_config= DataTransformation_config
    
    
    
    def get_data_transfomation(self):
        
        try:
            
           logging.info("Data transformation initaited")
            
           #define which col is  ordinal and ewhich is nminal
           catogrical_col=['cut', 'color', 'clarity']
           numerical_col=['carat', 'depth', 'table', 'x', 'y', 'z']   
        
        
        #define the custom ranking for eacjh oridnal  variable

           cut_category=["Fair","Good","Very Good" , "Premium" , "Ideal"]
           color_category = ["D","E","F","G","H","I","J"]
           clarity_category =["I1","SI2","SI1","VS2","VS1", "VVS2","VVS1","IF"]
           
           
           logging.info("Pipeline Initiated")
           
           #num pipeline
           num_pipeline= Pipeline(
    
                  steps= [
                          ("Imputer",SimpleImputer()),
       
                          ("Scaler",StandardScaler())   
                     ] 
    
                 )
           
           
           cat_pipeline=Pipeline(
    
    
                    steps=[
        
                         ("Imputer",SimpleImputer(strategy="most_frequent")) ,
                         ("Ordinal Encoding",OrdinalEncoder(categories=[cut_category,color_category,clarity_category]))
        
                     ]
    
    
                )
            
            
            
            #preproccesor
            
           preprosser=ColumnTransformer(
    
                            [
        
                      ("num_pipeline",num_pipeline,numerical_col),
                      ("cat_pipeline",cat_pipeline,catogrical_col)
        
        
                   ]
    
   
            )
           
           return preprosser
       
       
        except Exception as e:
            
             logging.info("Exception occur durin data transformation  stage")
             raise customexception(e,sys)
            
    
    
    
    
  
    def initiate_data_transformation(self ,traindata_path,testdata_path):
        
        try:
            
            train_df = pd.read_csv(traindata_path)
            
            test_df=pd.read_csv(testdata_path)
            
            logging.info("read traina nd test  data completed")
            logging.info(f"Train dataframe head :\n{train_df.head().to_string()}")
            logging.info(f"Train dataframe head :\n{test_df.head().to_string()}")
            
            
            preprossing_obj = self.get_data_transfomation()
            
            
            target_column_name = 'price'
            
            drop_column = [target_column_name,"id"]
            
            
            input_feature_train_df = train_df.drop(columns=drop_column ,axis=1)
            
            input_feature_test_df = test_df.drop(columns=drop_column)
            
            
            target_feature_train_df = train_df[target_column_name]
            
            target_feature_test_df = test_df[target_column_name]
            
            
            
            input_feature_train_arr = preprossing_obj.fit_transform(input_feature_train_df)
            
            input_feature_test_arr = preprossing_obj.transform(input_feature_test_df)
            
            
            train_arr = np.c_[input_feature_train_arr,np.array(target_feature_train_df)]
            
            test_arr = np.c_[input_feature_test_arr,np.array(target_feature_test_df)]
            
            logging.info("Applying  preprossing obj on train and test dataset")
            
            
            
            save_object(
                file_path = self.data_transformation_config.preprocessor_obj_file_path,
                
                obj = preprossing_obj
                
                
            )
            
            logging.info("object saved succesfully")
            
            return(
                
                train_arr,
                test_arr
                
            )
        
        
        
        except Exception as e:
            
            logging.info("Exception occur during data transformation  stage")
            raise customexception(e,sys)
            
                 
            
        