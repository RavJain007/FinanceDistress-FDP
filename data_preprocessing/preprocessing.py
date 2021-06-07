import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import power_transform
from sklearn.ensemble import ExtraTreesClassifier
from imblearn.over_sampling import ADASYN
import json
from Training_Raw_data_validation.rawValidation import Raw_Data_validation


class Preprocessor:

    def __init__(self, file_object, logger_object):
        self.file_object = file_object
        self.logger_object = logger_object
        self.schema_path = 'schema_training_train.json'

    def is_null_present(self, data):

        self.logger_object.log(self.file_object, 'Entered the is_null_present method of the Preprocessor class')
        self.null_present = False
        self.cols_with_missing_values = []
        self.cols = data.columns
        try:
            self.null_counts = data.isna().sum()  # check for the count of null values per column
            for i in range(len(self.null_counts)):
                if self.null_counts[i] > 0:
                    self.null_present = True
                    self.cols_with_missing_values.append(self.cols[i])
            if (self.null_present):  # write the logs to see which columns have null values
                self.dataframe_with_null = pd.DataFrame()
                self.dataframe_with_null['columns'] = data.columns
                self.dataframe_with_null['missing values count'] = np.asarray(data.isna().sum())
                self.dataframe_with_null.to_csv(
                    'preprocessing_data/null_values.csv')  # storing the null column information to file
            self.logger_object.log(self.file_object,
                                   'Finding missing values is a success.Data written to the null values file. Exited the is_null_present method of the Preprocessor class')
            return self.null_present, self.cols_with_missing_values
        except Exception as e:
            self.logger_object.log(self.file_object,
                                   'Exception occured in is_null_present method of the Preprocessor class. Exception message:  ' + str(
                                       e))
            self.logger_object.log(self.file_object,
                                   'Finding missing values failed. Exited the is_null_present method of the Preprocessor class')
            raise Exception()

    def impute_missing_values(self, data, cols_with_missing_values):

        self.logger_object.log(self.file_object, 'Entered the impute_missing_values method of the Preprocessor class')
        self.data = data
        self.cols_with_missing_values = cols_with_missing_values
        try:
            imputer=KNNImputer(n_neighbors=3, weights='uniform',missing_values=np.nan)
            self.new_array=imputer.fit_transform(self.data) # impute the missing values
            # convert the nd-array returned in the step above to a Dataframe
            self.new_data=pd.DataFrame(data=self.new_array, columns=self.data.columns)
            self.logger_object.log(self.file_object, 'Imputing missing values Successful. Exited the impute_missing_values method of the Preprocessor class')
            return self.new_data
        except Exception as e:
            self.logger_object.log(self.file_object,
                                   'Exception occured in impute_missing_values method of the Preprocessor class. Exception message:  ' + str(
                                       e))
            self.logger_object.log(self.file_object,
                                   'Imputing missing values failed. Exited the impute_missing_values method of the Preprocessor class')
            raise Exception()

    def scale_numerical_columns(self, data):

        self.logger_object.log(self.file_object,
                               'Entered the scale_numerical_columns method of the Preprocessor class')

        self.data = data

        try:
            self.num_df = self.data.select_dtypes(include=['int64']).copy()
            self.scaler = StandardScaler()
            self.scaled_data = self.scaler.fit_transform(self.num_df)
            self.scaled_num_df = pd.DataFrame(data=self.scaled_data, columns=self.num_df.columns)

            self.logger_object.log(self.file_object,
                                   'scaling for numerical values successful. Exited the scale_numerical_columns method of the Preprocessor class')
            return self.scaled_num_df

        except Exception as e:
            self.logger_object.log(self.file_object,
                                   'Exception occured in scale_numerical_columns method of the Preprocessor class. Exception message:  ' + str(
                                       e))
            self.logger_object.log(self.file_object,
                                   'scaling for numerical columns Failed. Exited the scale_numerical_columns method of the Preprocessor class')
            raise Exception()

    def separate_label_feature(self, data, label_column_name):

        self.logger_object.log(self.file_object, 'Entered the separate_label_feature method of the Preprocessor class')

        try:

            print(label_column_name)
            self.X = data.drop(labels=label_column_name,
                               axis=1)  # drop the columns specified and separate the feature columns

            self.Y = data[label_column_name] # Filter the Label columns
            self.logger_object.log(self.file_object,
                                   'Label Separation Successful. Exited the separate_label_feature method of the Preprocessor class')
            return self.X, self.Y

        except Exception as e:
            self.logger_object.log(self.file_object,
                                   'Exception occured in separate_label_feature method of the Preprocessor class                                                 Exceptionmessage:  ' + str(
                                       e))
            self.logger_object.log(self.file_object,
                                   'Label Separation Unsuccessful. Exited the separate_label_feature method of the                                               Preprocessor class')
            raise Exception()

    def label_encoder(self, data, column_name):

        self.logger_object.log(self.file_object,
                               'Entered the encode_categorical_columns method of the Preprocessor class')

        try:
            self.cat_df = data.select_dtypes(include=['object']).copy()
            # Using the dummy encoding to encode the categorical columns to numericsl ones
            for col in self.cat_df.columns:
                self.cat_df = LabelEncoder.fit_transform(self.cat_df, columns=[col], prefix=[col], drop_first=True)

            self.logger_object.log(self.file_object,
                                   'encoding for categorical values successful. Exited the encode_categorical_columns method of the Preprocessor class')
            return self.cat_df

        except Exception as e:
            self.logger_object.log(self.file_object,
                                   'Exception occured in encode_categorical_columns method of the Preprocessor class. Exception message:  ' + str(
                                       e))
            self.logger_object.log(self.file_object,
                                   'encoding for categorical columns Failed. Exited the encode_categorical_columns method of the Preprocessor class')
            raise Exception()

    def handle_imbalanced_dataset(self, x, y):

        self.logger_object.log(self.file_object,
                               'Entered the handle_imbalanced_dataset method of the Preprocessor class')

        try:
            #self.rdsmple = RandomOverSampler()
            #self.x_sampled, self.y_sampled = self.rdsmple.fit_sample(x, y)
            # transform the dataset
            self.oversample = ADASYN()
            self.x_sampled, self.y_sampled = self.oversample.fit_resample(x, y)
            self.logger_object.log(self.file_object,
                                   'dataset balancing successful. Exited the handle_imbalanced_dataset method of the Preprocessor class')

            return self.x_sampled, self.y_sampled

        except Exception as e:
            self.logger_object.log(self.file_object,
                                   'Exception occured in handle_imbalanced_dataset method of the Preprocessor class. Exception message:  ' + str(
                                       e))
            self.logger_object.log(self.file_object,
                                   'dataset balancing Failed. Exited the handle_imbalanced_dataset method of the Preprocessor class')
            raise Exception()

    def feature_selection(self, x):

        self.logger_object.log(self.file_object, 'Entered the feature_selection method')

        try:
            # Building the model


            '''
            def feature_selection(self, x, y, test):
            Below method used to find the train columns
            extra_tree_forest = ExtraTreesClassifier(n_estimators=5,
                                                     criterion='entropy', max_features=2)
            # Training the model
            extra_tree_forest.fit(x, y)
            # Computing the importance of each feature
            feature_importance = extra_tree_forest.feature_importances_
            # Normalizing the individual importance
            feature_importance_normalized= np.std([tree.feature_importances_ for tree in extra_tree_forest.estimators_], axis=0)
            ranked_features = pd.Series(extra_tree_forest.feature_importances_, index=x.columns)
            rf = ranked_features.nlargest(10)
            x = pd.DataFrame(data=x[rf.index])
            test = pd.DataFrame(data=test[rf.index]) '''
            with open(self.schema_path, 'r') as f:
                dic = json.load(f)
                f.close()
            column_names = dic['ColName']
            x = pd.DataFrame(data=x[column_names])

            return x

        except Exception as e:

            self.logger_object.log(self.file_object,
                                   'Exception occured in feature_selection method. Exception message:  ' + str(
                                       e))
            self.logger_object.log(self.file_object,
                                   'feature selection Failed. Exited the feature_selection method')
            raise Exception()

    def feature_scalar(self, data):

        self.logger_object.log(self.file_object, 'Entered the feature_scalar method')

        try:
            # transform training data
            self.data_columns = data.columns
            self.data = pd.DataFrame(power_transform(data, method='yeo-johnson'))
            self.data.columns = self.data_columns

            return self.data


        except Exception as e:

            self.logger_object.log(self.file_object,
                                   'Exception occured in feature_scalar method. Exception message:  ' + str(
                                       e))
            self.logger_object.log(self.file_object,
                                   'feature selection Failed. Exited the feature_scalar method')
            raise Exception()






