from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, accuracy_score
import mlflow
import mlflow.sklearn
# from hyperopt.pyll.base import scope
# from hyperopt import hp, fmin, tpe, Trials
from hyperopt import hp, fmin, tpe, Trials, STATUS_OK
# from sklearn.model_selection import RepeatedStratifiedKFold
import numpy as np
# from hyperopt.pyll import scope

class Model_Finder:
    """
                This class shall  be used to find the model with best accuracy and AUC score.
                Written By: iNeuron Intelligence
                Version: 1.0
                Revisions: None

                """

    def __init__(self, file_object, logger_object):
        self.file_object = file_object
        self.logger_object = logger_object
        self.clf = RandomForestClassifier()
        self.xgb = XGBClassifier(objective='binary:logistic')



    def get_best_params_for_random_forest(self, train_x, train_y, test_x, test_y):
        """
                                Method Name: get_best_params_for_random_forest
                                Description: get the parameters for Random Forest Algorithm which
                                             give the best accuracy.
                                             Use Hyper Parameter Tuning.
                                Output: The model with the best parameters
                                On Failure: Raise Exception


                                Version: 1.0
                                Revisions: None

                        """
       # self.logger_object.log(self.file_object, 'Entered the get_best_params_for_random_forest method of the Model_Finder class')
        try:
            # initializing with different combination of parameters
            space = {'criterion': hp.choice('criterion', ['entropy', 'gini']),
                     'max_depth': hp.quniform('max_depth', 10, 1200, 10),
                     'max_features': hp.choice('max_features', ['auto', 'sqrt', 'log2', None]),
                     'min_samples_leaf': hp.uniform('min_samples_leaf', 0, 0.5),
                     'min_samples_split': hp.uniform('min_samples_split', 0, 1),
                     'n_estimators' : hp.choice("n_estimators", [100, 200, 300, 400,500,600])
                     }

            def objective(space):
                model = RandomForestClassifier(criterion=space['criterion'],
                                               max_depth=space['max_depth'],
                                               max_features=space['max_features'],
                                               min_samples_leaf=space['min_samples_leaf'],
                                               min_samples_split=space['min_samples_split'],
                                               n_estimators=space['n_estimators'],
                                               )


                # log all the stuff here
                accuracy = cross_val_score(model, train_x, train_y, cv=4).mean()
                mlflow.log_metric('roc_auc_score',accuracy)
                model.fit(train_x, train_y)
                mlflow.sklearn.log_model(model, "model" )
                #modelpath = "model-%s-%f" % ("RF_", 1)
                #mlflow.sklearn.save_model(model, modelpath)

                return {'loss': -accuracy,"status":STATUS_OK}


            with mlflow.start_run():
                trials = Trials()
                self.best = fmin(fn=objective,
                            space=space,
                            algo=tpe.suggest,
                            max_evals=2,
                            trials=trials)


                #extracting the best parameters
                #self.criterion = self.best['criterion']
                print("Mrinal")
                #cr = ['entropy', 'gini']
                self.criterion = 'gini'
                self.max_depth = self.best['max_depth']
                # mx = ['auto', 'sqrt', 'log2', None]
                self.max_features = 'auto'
                self.n_estimators = self.best['n_estimators']
                self.min_samples_leaf = self.best['min_samples_leaf']
                self.min_samples_split = self.best['min_samples_split']



                #creating a new model with the best parameters
                self.clf = RandomForestClassifier(n_estimators=self.n_estimators, criterion=self.criterion,
                                                  max_depth=self.max_depth, max_features=self.max_features,
                                                  min_samples_leaf=self.min_samples_leaf,
                                                  min_samples_split=self.min_samples_split,
                                                  )
                # training the mew model
                self.clf.fit(train_x, train_y)
                self.logger_object.log(self.file_object,
                                   'Random Forest best params: ' + str(
                                       self.best) + '. Exited the get_best_params_for_random_forest method of the Model_Finder class')

                self.prediction_random_forest = self.clf.predict(test_x)  # prediction using the Random Forest Algorithm

                if len(test_y.unique()) == 1:  # if there is only one label in y, then roc_auc_score returns error. We will use accuracy in that case
                    self.random_forest_score = accuracy_score(test_y, self.clf)
                    self.logger_object.log(self.file_object, 'Accuracy for RF:' + str(self.clf))
                else:
                    self.random_forest_score = roc_auc_score(test_y,self.prediction_random_forest)  # AUC for Random Forest
                    self.logger_object.log(self.file_object, 'AUC for RF:' + str(self.random_forest_score))


                return self.random_forest_score, self.clf
        except Exception as e:
            self.logger_object.log(self.file_object,
                                   'Exception occured in get_best_params_for_random_forest method of the Model_Finder class. Exception message:  ' + str(
                                       e))
            self.logger_object.log(self.file_object,
                                   'Random Forest Parameter tuning  failed. Exited the get_best_params_for_random_forest method of the Model_Finder class')
            raise Exception()

    def get_best_params_for_random_forest_hyperopt(self, train_x, train_y, test_x, test_y):
        """
                                Method Name: get_best_params_for_random_forest
                                Description: get the parameters for Random Forest Algorithm which give the best accuracy.
                                             Use Hyper Parameter Tuning.
                                Output: The model with the best parameters
                                On Failure: Raise Exception


                                Version: 1.0
                                Revisions: None

                        """
        # self.logger_object.log(self.file_object, 'Entered the get_best_params_for_random_forest method of the Model_Finder class')
        try:
            mlflow.set_experiment('FDP')
            with mlflow.start_run(run_name='random_forest_hyperopt'):
                mlflow.log_param("dataset_shape", train_x.shape)
                mlflow.log_param("random_state", 29)

                def objective_function(params):
                    clf = RandomForestClassifier(**params, n_jobs=-1, random_state=29)
                    accuracies = cross_val_score(estimator=clf, X=train_x, y=train_y, cv=10, scoring='roc_auc')
                    loss_value = -1 * accuracies.mean()

                    return loss_value

                # initializing with different combination of parameters
                param_hyperopt_rf = {'criterion': hp.choice('criterion', ['entropy', 'gini']),
                                     # 'max_depth': hp.quniform('max_depth', 10, 1200, 10),
                                     'max_depth': hp.choice('max_depth', [int(x) for x in np.linspace(5, 30, num=6)]),
                                     'max_features': hp.choice('max_features', ['auto', 'sqrt', 'log2', None]),
                                     'min_samples_leaf': hp.uniform('min_samples_leaf', 0, 0.5),
                                     'min_samples_split': hp.uniform('min_samples_split', 0, 1),
                                     #  'n_estimators' : hp.choice("n_estimators", [100, 200, 300, 400,500,600])
                                     'n_estimators': hp.choice('n_estimators', [int(x) for x in
                                                                                np.linspace(start=1500, stop=5000,
                                                                                            num=12)]),
                                     }
              #  tpe = tpe.suggest
                tpe_trials = Trials()
                rf_bayesian_TPE = fmin(fn=objective_function,
                                       space=param_hyperopt_rf,
                                       algo=tpe.suggest,
                                       max_evals=2,
                                       trials=tpe_trials,
                                       rstate=np.random.RandomState(29))

                # Extract optimal values and parameter names:
                criterion = ['entropy', 'gini']
                max_features = ['auto', 'sqrt', 'log2', None]
                best_param_tpe = [x for x in rf_bayesian_TPE.values()]
                # param_names = [x for x in rf_bayesian_TPE.keys()]
                # Reset Random Forest with optimal parameters:
                param_hyperopt_rf['criterion'] = criterion[int(best_param_tpe[0])]
                param_hyperopt_rf['max_depth'] = int(best_param_tpe[1])
                param_hyperopt_rf['max_features'] = max_features[int(best_param_tpe[2])]
                param_hyperopt_rf['min_samples_leaf'] = float((best_param_tpe[3]))
                param_hyperopt_rf['min_samples_split'] = float(best_param_tpe[4])
                param_hyperopt_rf['n_estimators'] = int(best_param_tpe[5])
                self.logger_object.log(self.file_object, 'Best Parameter: Random Forest' + str(param_hyperopt_rf))
                print(param_hyperopt_rf)


                # creating a new model with the best parameters
                self.RF = RandomForestClassifier(n_estimators=param_hyperopt_rf['n_estimators'],
                                            max_depth=param_hyperopt_rf['max_depth'],
                                            max_features=param_hyperopt_rf['max_features'],
                                            min_samples_leaf=param_hyperopt_rf['min_samples_leaf'],
                                            min_samples_split=param_hyperopt_rf['min_samples_split'],
                                            criterion=param_hyperopt_rf['criterion'],
                                            class_weight='balanced'
                                            )
                    # training the mew model
                mlflow.log_params(param_hyperopt_rf)
                self.RF.fit(train_x, train_y)
                mlflow.sklearn.log_model(self.RF, "model")

                self.prediction_random_forest = self.RF.predict(test_x)  # prediction using the Random Forest Algorithm

                if len(
                        test_y.unique()) == 1:  # if there is only one label in y, then roc_auc_score returns error. We will use accuracy in that case
                    self.random_forest_score = accuracy_score(test_y, self.clf)
                    self.logger_object.log(self.file_object, 'Accuracy for RF:' + str(self.clf))
                else:
                    self.random_forest_score = roc_auc_score(test_y,
                                                             self.prediction_random_forest)  # AUC for Random Forest
                    self.logger_object.log(self.file_object, 'AUC for RF:' + str(self.random_forest_score))

                mlflow.log_metric("roc_auc_score", self.random_forest_score )
                mlflow.end_run()
                return self.random_forest_score, self.RF

        except Exception as e:
            self.logger_object.log(self.file_object,
                                   'Exception occured in get_best_params_for_random_forest method of the Model_Finder class. Exception message:  ' + str(
                                       e))
            self.logger_object.log(self.file_object,
                                   'Random Forest Parameter tuning  failed. Exited the get_best_params_for_random_forest method of the Model_Finder class')
            raise Exception()

    def get_best_params_for_xgboost(self,train_x,train_y,test_x,test_y):

        """
                                        Method Name: get_best_params_for_xgboost
                                        Description: get the parameters for XGBoost Algorithm which give the best accuracy.
                                                     Use Hyper Parameter Tuning.
                                        Output: The model with the best parameters
                                        On Failure: Raise Exception


                                        Version: 1.0
                                        Revisions: None

                                """
        self.logger_object.log(self.file_object,
                               'Entered the get_best_params_for_xgboost method of the Model_Finder class')
        try:
            # initializing with different combination of parameters
            self.param_grid_xgboost = {

                'learning_rate': [0.5, 0.1,],
                'max_depth': [3, 5, 10, 20],
                'n_estimators': [10, 50]

            }
            # Creating an object of the Grid Search class
            with mlflow.start_run():
                self.grid= GridSearchCV(XGBClassifier(objective='binary:logistic'),self.param_grid_xgboost, verbose=3,cv=5)
                # finding the best parameters
                self.grid.fit(train_x, train_y)

                # extracting the best parameters
                self.learning_rate = self.grid.best_params_['learning_rate']
                self.max_depth = self.grid.best_params_['max_depth']
                self.n_estimators = self.grid.best_params_['n_estimators']

                # creating a new model with the best parameters
                self.xgb = XGBClassifier(learning_rate=1, max_depth=5, n_estimators=50)
                # training the mew model
                self.xgb.fit(train_x, train_y)
                self.prediction_xgboost = self.xgb.predict(test_x)  # Predictions using the XGBoost Model

                if len(
                        test_y.unique()) == 1:  # if there is only one label in y, then roc_auc_score returns error. We will use accuracy in that case
                    self.xgboost_score = accuracy_score(test_y, self.prediction_xgboost)
                    self.logger_object.log(self.file_object,
                                           'Accuracy for XGBoost:' + str(self.xgboost_score))  # Log AUC
                else:
                    self.xgboost_score = roc_auc_score(test_y, self.prediction_xgboost)  # AUC for XGBoost
                    self.logger_object.log(self.file_object, 'AUC for XGBoost:' + str(self.xgboost_score))  # Log AUC

                mlflow.log_param("learning_rate", [0.5, 0.1,])
                mlflow.log_param("max_depth", [3, 5, 10, 20])
                mlflow.log_param("n_estimators", [10, 50])
                mlflow.log_metric("roc_auc_score", self.xgboost_score )
                mlflow.sklearn.log_model(self.xgb, "model")
                modelpath = "model-%s-%f" % ("XG_", 1)
                mlflow.sklearn.save_model(self.xgb , modelpath)
                self.logger_object.log(self.file_object,
                                       'XGBoost best params: ' + str(
                                           self.grid.best_params_) + '. Exited the get_best_params_for_xgboost method of the Model_Finder class')
                return self.xgboost_score, self.xgb
        except Exception as e:
            self.logger_object.log(self.file_object,
                                   'Exception occured in get_best_params_for_xgboost method of the Model_Finder class. Exception message:  ' + str(
                                       e))
            self.logger_object.log(self.file_object,
                                   'XGBoost Parameter tuning  failed. Exited the get_best_params_for_xgboost method of the Model_Finder class')
            raise Exception()

    def get_best_params_for_xgboost_hyperopt(self, train_x, train_y, test_x, test_y):

        """
                                        Method Name: get_best_params_for_xgboost
                                        Description: get the parameters for XGBoost Algorithm which give the best accuracy.
                                                     Use Hyper Parameter Tuning.
                                        Output: The model with the best parameters
                                        On Failure: Raise Exception


                                        Version: 1.0
                                        Revisions: None

                                """
        self.logger_object.log(self.file_object,
                               'Entered the get_best_params_for_xgboost method of the Model_Finder class')
        try:
            # initializing with different combination of parameters
            mlflow.set_experiment('FDP')
            with mlflow.start_run(run_name='XGBoost_hyperopt'):
                mlflow.log_param("dataset_shape", train_x.shape)

                space = {
                    'max_depth': hp.choice('max_depth', [int(x) for x in np.linspace(5, 30, num=6)]),
                    'learning_rate': hp.quniform('learning_rate', 0.01, 0.6, 0.01),
                    'n_estimators': hp.choice('n_estimators', [int(x) for x in np.linspace(start=1500, stop=5000, num=12)]),
                    'gamma': hp.quniform('gamma', 0, 5, 0.01),
                    'min_child_weight': hp.quniform('min_child_weight', 1, 10, 1),
                    'subsample': hp.quniform('subsample', 0.1, 1, 0.01),
                    'colsample_bytree': hp.quniform('colsample_bytree', 0.1, 1.0, 0.01)
                }
                def objective(space):

                    classifier = XGBClassifier(n_estimators=space['n_estimators'],
                                               max_depth=int(space['max_depth']),
                                               learning_rate=space['learning_rate'],
                                               gamma=space['gamma'],
                                               min_child_weight=space['min_child_weight'],
                                               subsample=space['subsample'],
                                               colsample_bytree=space['colsample_bytree']
                                               )



                    classifier.fit(train_x, train_y)

                    # Applying k-Fold Cross Validation
                    from sklearn.model_selection import cross_val_score
                    accuracies = cross_val_score(estimator=classifier, X=train_x, y=train_y, cv=10, scoring='roc_auc')
                    CrossValMean = accuracies.mean()

                    print("CrossValMean:", CrossValMean)

                    return {'loss': 1 - CrossValMean, 'status': STATUS_OK}


                # Creating an object of the Grid Search class

                trials = Trials()
                best = fmin(fn=objective,
                            space=space,
                            algo=tpe.suggest,
                            max_evals=2,
                            trials=trials)

                print("Best: ", best)
                self.logger_object.log(self.file_object, 'Best Parameter: XGBoost Forest' + str(best))

                # creating a new model with the best parameters

                self.xgb = XGBClassifier(n_estimators=best['n_estimators'],
                                           max_depth=best['max_depth'],
                                           learning_rate=best['learning_rate'],
                                           gamma=best['gamma'],
                                           min_child_weight=best['min_child_weight'],
                                           subsample=best['subsample'],
                                           colsample_bytree=best['colsample_bytree']
                                           )

                mlflow.log_params(best)
                self.xgb.fit(train_x, train_y)
                mlflow.sklearn.log_model(self.xgb, "model")

                self.prediction_xgboost = self.xgb.predict(test_x)  # Predictions using the XGBoost Model

                if len(
                        test_y.unique()) == 1:  # if there is only one label in y, then roc_auc_score returns error. We will use accuracy in that case
                    self.xgboost_score = accuracy_score(test_y, self.prediction_xgboost)
                    self.logger_object.log(self.file_object,
                                           'Accuracy for XGBoost:' + str(self.xgboost_score))  # Log AUC
                else:
                    self.xgboost_score = roc_auc_score(test_y, self.prediction_xgboost)  # AUC for XGBoost
                    self.logger_object.log(self.file_object, 'AUC for XGBoost:' + str(self.xgboost_score))  # Log AUC

                mlflow.log_metric("roc_auc_score", self.xgboost_score )
                mlflow.end_run()

                return self.xgboost_score, self.xgb
        except Exception as e:
            self.logger_object.log(self.file_object,
                                   'Exception occurred in get_best_params_for_xgboost '
                                   'method of the Model_Finder class. Exception message:  ' + str(e))
            self.logger_object.log(self.file_object,
                                   'XGBoost Parameter tuning  failed. '
                                   'Exited the get_best_params_for_xgboost method of the Model_Finder class')
            raise Exception()

    def get_best_model(self,train_x, train_y, test_x, test_y):
        """
                                                Method Name: get_best_model
                                                Description: Find out the Model which has the best AUC score.
                                                Output: The best model name and the model object
                                                On Failure: Raise Exception

                                                Written By: iNeuron Intelligence
                                                Version: 1.0
                                                Revisions: None

                                        """
        self.logger_object.log(self.file_object,
                               'Entered the get_best_model method of the Model_Finder class')
        # create best model for XGBoost
        try:
            self.xgboost_score, self.xgboost = \
                self.get_best_params_for_xgboost_hyperopt(train_x, train_y, test_x, test_y)
            self.random_forest_score, self.random_forest = \
                self.get_best_params_for_random_forest_hyperopt(train_x, train_y, test_x, test_y)
            # comparing the two models
            if self.random_forest_score < self.xgboost_score:
                return 'XGBoost', self.xgboost
            else:
                return 'RandomForest', self.random_forest

        except Exception as e:
            self.logger_object.log(self.file_object,
                                   'Exception occured in get_best_model method of the Model_Finder '
                                   'class. Exception message:  ' + str(e))
            self.logger_object.log(self.file_object,
                                   'Model Selection Failed. Exited the get_best_model method of the Model_Finder class')
            raise Exception()
