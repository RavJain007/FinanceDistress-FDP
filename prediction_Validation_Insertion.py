from datetime import datetime
import pandas as pd
from Prediction_Raw_Data_Validation.predictionDataValidation import Prediction_Data_validation
from DataTypeValidation_Insertion_Prediction.DataTypeValidationPrediction import dBOperation

from application_logging import logger

class pred_validation:
    def __init__(self,path,Batch):
        self.Batch = Batch
        if self.Batch == 'Batch':
            self.raw_data = Prediction_Data_validation(path,Batch)
        elif self.Batch == 'UI':
            self.Df = path
        elif self.Batch == 'UP':
            self.Df = path
            self.raw_data = Prediction_Data_validation(path, Batch)

        self.dBOperation = dBOperation()
        self.file_object = open("Prediction_Logs/Prediction_Log.txt", 'a+')
        self.log_writer = logger.App_Logger()

    def prediction_validation(self):

        try:
            if self.Batch == 'Batch':
                self.log_writer.log(self.file_object, 'Start of Validation on files for prediction!!')
                # extracting values from prediction schema
                pattern, column_names, noofcolumns = self.raw_data.valuesFromSchema()
                # getting the regex defined to validate filename
                regex = self.raw_data.manualRegexCreation()
                # validating filename of prediction files
                self.raw_data.validationFileNameRaw(regex)
                # validating column length in the file
                self.raw_data.validateColumnLength(noofcolumns)
                # self.log_writer.log(self.file_object, "Raw Data Validation Complete!!")
                self.log_writer.log(self.file_object,
                                    "Creating Prediction_Database and tables on the basis of given schema!!!")
                # create database with given name, if present open the connection! Create table with columns given in schema
                self.dBOperation.createTableDb()
                self.log_writer.log(self.file_object, "Collection creation Completed!!")
                self.log_writer.log(self.file_object, "Insertion of Data into Collection started!!!!")
                # insert csv files in the table
                self.dBOperation.insertIntoTableGoodData()
                self.log_writer.log(self.file_object, "Insertion in Collection completed!!!")
                self.log_writer.log(self.file_object, "Deleting Good Data Folder!!!")
                # Delete the good data folder after loading files in tale
                self.raw_data.deleteExistingGoodDataTrainingFolder()
                self.log_writer.log(self.file_object, "Good_Data folder deleted!!!")
                self.log_writer.log(self.file_object, "Moving bad files to Archive and deleting Bad_Data folder!!!")
                # Move the bad files to archive folder
                self.raw_data.moveBadFilesToArchiveBad()
                self.log_writer.log(self.file_object, "Bad files moved to archive!! Bad folder Deleted!!")
                self.log_writer.log(self.file_object, "Validation Operation completed!!")
                self.log_writer.log(self.file_object, "Extracting csv file from table")
                # export data in table to csvfile
                self.dBOperation.selectingDatafromtableintocsv()
                self.file_object.close()
            elif self.Batch == 'UP':
                self.log_writer.log(self.file_object, 'Start of Validation on files for prediction!!')
                # extracting values from prediction schema
                pattern, column_names, noofcolumns = self.raw_data.valuesFromSchema()
                if self.Df .shape[1] == noofcolumns:
                    self.log_writer.log(self.file_object, 'Column Length Validation Completed!!!!')
                else:
                    self.log_writer.log(self.file_object, 'Column Length Validation Failed!!!!')
                self.log_writer.log(self.file_object, "Raw Data Validation Complete!!")

            elif self.Batch == 'UI':
                self.log_writer.log(self.file_object, 'Start of Validation on files for prediction!!')
                # extracting values from prediction schema




        except Exception as e:
            raise e









