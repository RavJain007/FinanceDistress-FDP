import shutil
# import sqlite3
from datetime import datetime
from scipy.io.arff import loadarff
from os import listdir
import os
import csv
#import arff
import pymongo
import pandas as pd
from pathlib import Path


from application_logging.logger import App_Logger

class dBOperation:
    """
      This class shall be used for handling all the SQL operations.
      Written By: Vijeta
      Version: 1.0
      Revisions: None
      """

    def __init__(self):
        self.path = 'Prediction_Database/'
        self.badFilePath = "Prediction_Raw_Files_Validated/Bad_Raw"
        self.goodFilePath = "Prediction_Raw_Files_Validated/Good_Raw"
        self.logger = App_Logger()

    def dataBaseConnection(self):
        """
                Method Name: dataBaseConnection
                Description: This method creates the database with the given name and if Database already exists then opens the connection to the DB.
                Output: Connection to the DB
                Version: 1.0
                Revisions: None
                """
        try:
            connection_url = 'mongodb+srv://vijetan:12345@cluster0.lvv35.mongodb.net/<Test1>?retryWrites=true&w=majority'
            client = pymongo.MongoClient(connection_url)
            # conn = sqlite3.connect(self.path+DatabaseName+'.db')

            file = open("Prediction_Logs/DataBaseConnectionLog.txt", 'a+')
            self.logger.log(file, "Opened %s database successfully")
            file.close()
        except ConnectionError:
            file = open("Prediction_Logs/DataBaseConnectionLog.txt", 'a+')
            self.logger.log(file, "Error while connecting to database: %s" % ConnectionError)
            file.close()
            raise ConnectionError
        return client

    def createTableDb(self):
        """
                        Method Name: createTableDb
                        Description: This method creates a table in the given database which will be used to insert the Good data after raw data validation.
                        Output: None
                        On Failure: Raise Exception
                         Written By: Vijeta
                        Version: 1.0
                        Revisions: None
                        """
        try:
            client = self.dataBaseConnection()

            database = client.FinancialDistress
            sampletable = database["Prediction"]

            print("Table created")

            # c=client.cursor()
            # c.execute("SELECT count(name)  FROM sqlite_master WHERE type = 'table'AND name = 'Good_Raw_Data'")
            file = open("Prediction_Logs/DbTableCreateLog.txt", 'a+')
            self.logger.log(file, "Tables created successfully!!")
            file.close()

            file = open("Prediction_Logs/DataBaseConnectionLog.txt", 'a+')
            self.logger.log(file, "Closed %s database successfully")
            file.close()
        except Exception as e:
            file = open("Prediction_Logs/DbTableCreateLog.txt", 'a+')
            self.logger.log(file, "Error while creating table: %s " % e)
            file.close()
            file = open("Prediction_Logs/DataBaseConnectionLog.txt", 'a+')
            self.logger.log(file, "Closed %s database successfully")
            file.close()
            raise e
        return sampletable

    def insertIntoTableGoodData(self):

        """
                               Method Name: insertIntoTableGoodData
                               Description: This method inserts the Good data files from the Good_Raw folder into the
                                            above created table.
                               Output: None
                               On Failure: Raise Exception

                                Written By: iNeuron Intelligence
                               Version: 1.0
                               Revisions: None

        """
        # data = loadarff(r'Training_Raw_files_validated\Good_Raw\1year.arff')
        # print("Data Loaded")
        client = self.dataBaseConnection()
        # client.drop_database(name_or_database="FinancialDistress")

        # print("Dropped data 1")
        goodFilePath = self.goodFilePath
        badFilePath = self.badFilePath
        onlyfiles = [f for f in listdir(goodFilePath)]
        log_file = open("Prediction_Logs/DbInsertLog.txt", 'a+')
        for file in onlyfiles:
            try:
                mydb = client["FinancialDistress"]
                mycol = mydb["Prediction"]
                print("file is ", file)
                x = mycol.delete_many({"filename": file})

                print(x.deleted_count, " documents deleted.")
                f = goodFilePath + '/' + file
                path1 = Path(f)
                print(path1)
                data = loadarff(path1)
                df = pd.DataFrame(data[0])
                print(file)
                df['filename'] = os.path.basename(path1)
                print(df)
                data_dict = df.to_dict("records")
                sampletable = self.createTableDb()
                query = sampletable.insert_many(data_dict)

            except Exception as e:

                raise e
                self.logger.log(log_file, "Error while creating table: %s " % e)
                print("Exception")
                shutil.move(goodFilePath + '/' + file, badFilePath)
                self.logger.log(log_file, "File Moved Successfully %s" % file)
                log_file.close()

    def selectingDatafromtableintocsv(self):

        """
                               Method Name: selectingDatafromtableintocsv
                               Description: This method exports the data in GoodData table as a CSV file. in a given location.
                                            above created .
                               Output: None
                               On Failure: Raise Exception
                               Revisions: None
        """

        self.fileFromDb = 'Prediction_FileFromDB/'
        self.fileName = 'InputFile.csv'
        log_file = open("Prediction_Logs/ExportToCsv.txt", 'a+')
        try:
            client = self.dataBaseConnection()
            database = client.FinancialDistress
            sampletable = database.Prediction
            cursor = sampletable.find()

            #print("Fetching Data")
            if not os.path.isdir(self.fileFromDb):
                    os.makedirs(self.fileFromDb)
            mongo_docs = list(cursor)
            print("list")
            mongo_docs = mongo_docs[:]
            # create an empty DataFrame obj for storing Series objects
            #docs = pd.DataFrame(columns=[])
            docs= pd.DataFrame(mongo_docs)
            """for num, doc in enumerate(mongo_docs):
                print("loop")
                doc["_id"] = str(doc["_id"])
                doc_id = doc["_id"]
                series_obj = pd.Series(doc, name=doc_id)
                docs = docs.append(series_obj)"""

            docs.to_csv(self.fileFromDb + self.fileName, ",",index=False)



            self.logger.log(log_file, "File exported successfully!!!")
            log_file.close()

        except Exception as e:
            self.logger.log(log_file, "File exporting failed. Error : %s" % e)
            log_file.close()