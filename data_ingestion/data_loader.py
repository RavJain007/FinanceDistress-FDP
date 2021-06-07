import pandas as pd
from sklearn.preprocessing import LabelEncoder

class Data_Getter:
    """
    This class shall  be used for obtaining the data from the source for training.

    Written By: Saurabh Purohit
    Version: 1.0
    Revisions: None

    """
    def __init__(self, file_object, logger_object):
        self.training_file='Training_FileFromDB/InputFile.csv'
        self.file_object=file_object
        self.logger_object=logger_object

    def get_data(self):
        """
        Method Name: get_data
        Description: This method reads the data from source.
        Output: A pandas DataFrame.
        On Failure: Raise Exception

         Written By: Saurabh Purohit
        Version: 1.0
        Revisions: None

        """
        self.logger_object.log(self.file_object,'Entered the get_data method of the Data_Getter class')
        try:
            self.data= pd.read_csv(self.training_file) # reading the data file
            #le = LabelEncoder()
            #self.data['class'] = le.fit_transform(self.data['class'])
            self.data['class'] = self.data['class'].replace("b'0'", 0)
            self.data['class'] = self.data['class'].replace("b'1'", 1)
            #self.data.astype('float')
            self.data.drop('_id', axis=1, inplace=True)
            self.data.drop('filename', axis=1, inplace=True)
            #self.data.convert_objects(convert_numeric=True).dtypes
            cols = self.data.columns.drop('class')
            self.data[cols] = self.data[cols].apply(pd.to_numeric, errors='coerce')
            self.data['class'] = self.data['class'].astype('int')
            self.logger_object.log(self.file_object,'Data Load Successful.Exited the get_data method of the Data_Getter class')
            self.data = self.data[self.data.duplicated() == False] # remove if any duplicate

            return self.data
        except Exception as e:
            self.logger_object.log(self.file_object,'Exception occured in get_data method of the Data_Getter class. Exception message: '+str(e))
            self.logger_object.log(self.file_object,
                                   'Data Load Unsuccessful.Exited the get_data method of the Data_Getter class')
            raise Exception()