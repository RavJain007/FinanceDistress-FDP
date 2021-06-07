import json

import pandas
from file_operations import file_methods
from data_preprocessing import preprocessing_pred
from data_ingestion import data_loader_prediction
from application_logging import logger
from Prediction_Raw_Data_Validation.predictionDataValidation import Prediction_Data_validation
import pandas as pd


class prediction:

    def __init__(self,path,Batch):
        self.file_object = open("Prediction_Logs/Prediction_Log.txt", 'a+')
        self.log_writer = logger.App_Logger()
        self.Batch = Batch
        if self.Batch == 'Batch':
            self.pred_data_val = Prediction_Data_validation(path,self.Batch)
        elif self.Batch == 'UI':
            self.Df = path
        elif self.Batch == 'UP':
            self.Df = path
            self.pred_data_val = Prediction_Data_validation(path, self.Batch)

    def predictionFromModel(self):

        try:
            if self.Batch == 'Batch':
                self.pred_data_val.deletePredictionFile() #deletes the existing prediction file from last run!
                self.log_writer.log(self.file_object,'Start of Prediction')
                data_getter=data_loader_prediction.Data_Getter_Pred(self.file_object,self.log_writer)
                data=data_getter.get_data()

                #code change

                preprocessor=preprocessing_pred.Preprocessor(self.file_object,self.log_writer)

                # check if missing values are present in the dataset
                is_null_present, cols_with_missing_values = preprocessor.is_null_present(data)
                if (is_null_present):
                    data = preprocessor.impute_missing_values(data, cols_with_missing_values)  # missing value imputation
                data = preprocessor.feature_scalar(data)
                data = preprocessor.feature_selection(data)

                # get encoded values for categorical data


                #data=data.to_numpy()
                file_loader=file_methods.File_Operation(self.file_object,self.log_writer)

                file_loader = file_methods.File_Operation(self.file_object, self.log_writer)
                model_name = file_loader.find_model_file()
                model = file_loader.load_model(model_name)
                result = pd.DataFrame(model.predict(data))
                result.to_csv("Prediction_Output_File/Predictions.csv", header=True,
                              mode='a+')  # appends result to prediction
                self.log_writer.log(self.file_object,'End of Prediction')

                path = "Predicting completed please"

            else:
                preprocessor = preprocessing_pred.Preprocessor(self.file_object, self.log_writer)
                is_null_present, cols_with_missing_values = preprocessor.is_null_present(self.Df)
                if (is_null_present):
                    data = preprocessor.impute_missing_values(self.Df,
                                                              cols_with_missing_values)  # missing value imputation
                else:
                    data = self.Df
                data = preprocessor.feature_scalar(data)
                if self.Batch == 'UI':
                    with open('schema_training_train.json', 'r') as f:
                        dic = json.load(f)
                        f.close()
                    column_names = dic['ColName']
                    data.columns = column_names

                if self.Batch == 'UP':
                    data = preprocessor.feature_selection(data)
                file_loader = file_methods.File_Operation(self.file_object, self.log_writer)
                model_name = file_loader.find_model_file()
                model = file_loader.load_model(model_name)
                result = pd.DataFrame(model.predict(data))
                result.to_csv("Prediction_Output_File/Predictions.csv", header=True,
                              mode='a+')  # appends result to predictio
                if self.Batch == 'UP':
                    path = result.to_json()
                else:
                    if result.iloc[0][0] == 0:
                        path = "The Company is not Financially Distress"
                    else:
                        path = "The Company is a Financially Distress"


                print(path)
                self.log_writer.log(self.file_object, 'End of Prediction')

        except Exception as ex:
            self.log_writer.log(self.file_object, 'Error occured while running the prediction!! Error:: %s' % ex)
            raise ex
        return path





