from DimondPricePrediction.pipelines.prediction_pipeline import customdata


custom_obj = customdata(0.3,62.0,56.0,4.35,4.37,2.7,"Ideal","F","IF")

data= custom_obj.get_data_as_df()

print(data)