# import json

from sklearn.model_selection import train_test_split
from data_ingestion import data_loader
from data_preprocessing import preprocessing
# from data_preprocessing import clustering
from best_model_finder import tuner
# from file_operations import file_methods
from application_logging import logger
# import pandas as pd
import pickle
import os
import shutil
import mlflow

class trainModel:

    def __init__(self):
        self.log_writer = logger.App_Logger()
        self.file_object = open("Training_Logs/ModelTrainingLog.txt", 'a+')

    def trainingModel(self):
        try:

            data_getter = data_loader.Data_Getter(self.file_object, self.log_writer)
            data = data_getter.get_data()
            """doing the data preprocessing"""
            preprocessor = preprocessing.Preprocessor(self.file_object, self.log_writer)
            # data.replace('?',np.NaN,inplace=True) # replacing '?' with NaN values for imputation
            # create separate features and labels
            X, Y = preprocessor.separate_label_feature(data, label_column_name='class')

            # check if missing values are present in the dataset
            is_null_present, cols_with_missing_values = preprocessor.is_null_present(X)
            # if missing values are there, replace them appropriately.
            if is_null_present:
                X = preprocessor.impute_missing_values(X, cols_with_missing_values)  # missing value imputation

            # Log in MLFLOW

            x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=1 / 3, random_state=355)
            # Applying yeo-johnson scalar
            x_train = preprocessor.feature_scalar(x_train)
            x_test = preprocessor.feature_scalar(x_test)

            # Feature Selection

            x_train = preprocessor.feature_selection(x_train)
            x_test = preprocessor.feature_selection(x_test)


            # Handling imbalance data
            x_train, y_train = preprocessor.handle_imbalanced_dataset(x_train, y_train)

            model_finder = tuner.Model_Finder(self.file_object, self.log_writer)  # object initialization
            # getting the best model for each of the clusters
            best_model_name, best_model = model_finder.get_best_model(x_train, y_train, x_test, y_test)
            model_directory = 'models/'
            if os.path.isdir(model_directory):  # remove previously existing models
                shutil.rmtree(model_directory)
                os.makedirs(model_directory)

            path = os.path.join(model_directory, best_model_name)  # create seperate  directory for each cluster
            if os.path.isdir(path):  # remove previously existing models for each clusters
                shutil.rmtree(model_directory)
                os.makedirs(path)
            else:
                os.makedirs(path)  #
            with open(path + '/' + best_model_name + '.sav',
                      'wb') as f:
                pickle.dump(best_model, f)  # save the model to file
            self.log_writer.log(self.file_object, 'Model File ' + best_model_name +
                                ' saved. Exited the save_model method of the Model_Finder class')
            # logging the successful Training
            self.log_writer.log(self.file_object, 'Successful End of Training')
            self.file_object.close()
            result = "Training is successful and model is saved"
            return result

        except Exception:
            # logging the unsuccessful Training
            self.log_writer.log(self.file_object, 'Unsuccessful End of Training')
            self.file_object.close()
            raise Exception
