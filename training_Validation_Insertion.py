# from datetime import datetime

# from DataTypeValidation_Insertion_Training.DataTypeValidation import dBOperation
from Training_Raw_data_validation.rawValidation import Raw_Data_validation
from DataTypeValidation_Insertion_Training.DataTypeValidation import dBOperation

from application_logging import logger


class train_validation:
    def __init__(self,path,Batch):
        self.Batch = Batch
        #if self.Batch == 'Batch':
        self.raw_data = Raw_Data_validation(path, self.Batch)
        if self.Batch == 'UI':
            self.Df = path


        self.dBOperation = dBOperation()
        self.file_object = open("Training_Logs/Training_Main_Log.txt", 'a+')

        self.log_writer = logger.App_Logger()

    def train_validation(self):
        try:
            if self.Batch == 'Batch':
                self.log_writer.log(self.file_object, 'Start of Validation on files for Training!!')
                # extracting values from training schema
                pattern,column_names, noofcolumns = self.raw_data.valuesFromSchema()
                # getting the regex defined to validate filename
                regex = self.raw_data.manualRegexCreation()
                # validating filename of training files
                self.raw_data.validationFileNameRaw(regex)
                # validating column length in the file
                self.raw_data.validateColumnLength(noofcolumns)
                self.log_writer.log(self.file_object, "Raw Data Validation Complete!!")
                self.log_writer.log(self.file_object,"Creating Training_Database and tables on the basis of given schema!!!")
                #create database with given name, if present open the connection! Create table with columns given in schema
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
            else:
                self.log_writer.log(self.file_object, 'Start of Validation on files for Training!!')
                # extracting values from training schema
                pattern,column_names, noofcolumns = self.raw_data.valuesFromSchema()


                # validating column length in the file

                if self.Df .shape[1] == noofcolumns:
                    self.log_writer.log(self.file_object, 'Column Length Validation Completed!!!!')
                else:
                    self.log_writer.log(self.file_object, 'Column Length Validation Failed!!!!')

                self.log_writer.log(self.file_object, "Raw Data Validation Complete!!")
                self.log_writer.log(self.file_object,
                                    "Creating Training_Database and tables on the basis of given schema!!!")
                # create database with given name, if present open the connection! Create table with columns given in schema
                self.dBOperation.createTableDb()
                self.log_writer.log(self.file_object, "Collection creation Completed!!")
                self.log_writer.log(self.file_object, "Insertion of Data into Collection started!!!!")
                # insert csv files in the table
                self.dBOperation.insertIntoTableUI(self.Df)
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

        except Exception as e:
            raise e









