from cmath import log
import importlib
from pyexpat import model
import numpy as np
import yaml
import sys,os
from collections import namedtuple
from typing import List

from housing.exception import HousingException
from housing.logger import logging
from sklearn.metrics import r2_score,mean_squared_error

GRID_SEARCH_KEY = 'grid_search'
MODULE_KEY = 'module'
CLASS_KEY = 'class'
PARAM_KEY = 'params'
MODEL_SELECTION_KEY = 'model_selection'
SEARCH_PARAM_GRID_KEY = "search_param_grid"


InitializedModelDetail = namedtuple('InitializedModelDetail',
                                    ['model_serial_number','model','param_grid_search','model_name'])

GridSearchedBestModel = namedtuple('GridSearchedBestModel',
                                   ['model_serial_number',
                                    'model',
                                    'best_model',
                                    'best_parameters',
                                    'best_score'
                                    ])
BestModel = namedtuple("BestModel", ["model_serial_number",
                                     "model",
                                     "best_model",
                                     "best_parameters",
                                     "best_score", ])


MetricInfoArtifact = namedtuple("MetricInfoArtifact",
                                ["model_name","model_object","train_rmse","test_rmse","train_accuracy",
                                 "test_accuracy","model_accuracy","index_number"])

#Model object is tained model object and model accuracy means average of training and testing accuracy
def evaluate_regression_model(model_list: list, X_train:np.ndarray, y_train:np.ndarray, X_test:np.ndarray, y_test:np.ndarray, base_accuracy:float=0.6) -> MetricInfoArtifact:
    """
    Description:
    This function compare multiple regression model return best model

    Params:
    model_list: List of model
    X_train: Training dataset input feature
    y_train: Training dataset target feature
    X_test: Testing dataset input feature
    y_test: Testing dataset input feature

    return
    It retured a named tuple
    
    MetricInfoArtifact = namedtuple("MetricInfo",
                                ["model_name", "model_object", "train_rmse", "test_rmse", "train_accuracy",
                                 "test_accuracy", "model_accuracy", "index_number"])

    """
    try:
        
        #model_list = [model.best_model for model in model_list]
        index_number = 0
        metric_info_artifact = None
        for model in model_list:
            model_name = str(model)  #getting model name based on model object
            logging.info(f"{'>>'*30}Started evaluating model: [{type(model).__name__}] {'<<'*30}")
            
            #Getting prediction for training and testing dataset
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            #Calculating r squared score on training and testing dataset
            train_acc = r2_score(y_train, y_train_pred)
            test_acc = r2_score(y_test, y_test_pred)
            
            #Calculating mean squared error on training and testing dataset
            train_rmse = mean_squared_error(y_train, y_train_pred)
            test_rmse = mean_squared_error(y_test, y_test_pred)

            # Calculating harmonic mean of train_accuracy and test_accuracy
            model_accuracy = (2 * (train_acc * test_acc)) / (train_acc + test_acc)
            diff_test_train_acc = abs(test_acc - train_acc)
            
            #logging all important metric
            logging.info(f"{'>>'*30} Score {'<<'*30}")
            logging.info(f"Train Score\t\t Test Score\t\t Average Score")
            logging.info(f"{train_acc}\t\t {test_acc}\t\t{model_accuracy}")

            logging.info(f"{'>>'*30} Loss {'<<'*30}")
            logging.info(f"Diff test train accuracy: [{diff_test_train_acc}].") 
            logging.info(f"Train root mean squared error: [{train_rmse}].")
            logging.info(f"Test root mean squared error: [{test_rmse}].")


            #if model accuracy is greater than base accuracy and train and test score is within certain thershold
            #we will accept that model as accepted model
            if model_accuracy >= base_accuracy and diff_test_train_acc < 0.05:
                base_accuracy = model_accuracy
                metric_info_artifact = MetricInfoArtifact(model_name=model_name,
                                                        model_object=model,
                                                        train_rmse=train_rmse,
                                                        test_rmse=test_rmse,
                                                        train_accuracy=train_acc,
                                                        test_accuracy=test_acc,
                                                        model_accuracy=model_accuracy,
                                                        index_number=index_number)

                logging.info(f"Acceptable model found {metric_info_artifact}. ")
            index_number += 1
        if metric_info_artifact is None:
            logging.info(f"No model found with higher accuracy than base accuracy")
        
        logging.info(f"Metric Info Artifact is: {metric_info_artifact}")
        return metric_info_artifact
    except Exception as e:
        raise HousingException(e, sys) from e




class ModelFactory:

    def __init__(self,model_config_path:str = None):
            try:
                self.config:dict = ModelFactory.read_params(model_config_path)
                self.grid_search_cv_module: str = self.config[GRID_SEARCH_KEY][MODULE_KEY]
                self.grid_search_class_name: str = self.config[GRID_SEARCH_KEY][CLASS_KEY]
                self.grid_search_property_data:dict = dict(self.config[GRID_SEARCH_KEY][PARAM_KEY])

                self.models_initialization_config:dict = dict(self.config[MODEL_SELECTION_KEY])

                self.initialized_model_list = None
                self.grid_searched_best_model_list = None

            except Exception as e:
                    raise HousingException(e,sys) from e
            
    @staticmethod
    def read_params(config_path:str)->dict:
        try:
            with open(config_path) as yaml_file:
                  config:dict = yaml.safe_load(yaml_file)

            return config

        except Exception as e:
                raise HousingException(e,sys) from e 

    @staticmethod
    def get_class_and_module(module_name:str,class_name:str):
        try:
            #load the module using imoprtlib, in case it failed to load , it will raise ImportError
            module = importlib.import_module(module_name)
            # get the class, will raise AttributeError if class cannot be found
            logging.info(f"Executing command from {module} import {class_name}")
            
            class_ref = getattr(module,class_name)

            return class_ref
        
        except Exception as e:
                raise HousingException(e,sys) from e
        



    @staticmethod
    def update_property_of_class(instance_ref:object,property_data:dict):
        try:
            if not isinstance(property_data,dict):
                    raise Exception("Property_data parameter required as dictionary format")

            print(property_data)
            for key,value in property_data.items():
                logging.info(f"Executing:$ {str(instance_ref)}.{key}={value}")
                setattr(instance_ref,key,value)

            return instance_ref
        
        except Exception as e:
              raise HousingException(e,sys) from e
        



    def get_initialised_model_list(self)->List[InitializedModelDetail] :
        """
        This function will return a list of tupels(model details).
        return List[ModelDetail]
        """   
        try:
            initialized_model_list = []       
            for model_serial_number in self.models_initialization_config.keys():
                  
                #To get model details by its serial number
                model_initailization_config = self.models_initialization_config[model_serial_number]
                
                #import class reference from module
                model_obj_ref = ModelFactory.get_class_and_module(model_initailization_config[MODULE_KEY],
                                                                model_initailization_config[CLASS_KEY])
                
                #create model class
                model = model_obj_ref()
            
                # To check if there is parameters dictionary present 
                if PARAM_KEY in model_initailization_config:
                    model_obj_property_data = dict(model_initailization_config[PARAM_KEY])

                    # Update property of model class by using given values for parameters
                    model = ModelFactory.update_property_of_class(instance_ref=model,
                                                                    property_data=model_obj_property_data)
                
                param_grid_search = model_initailization_config[SEARCH_PARAM_GRID_KEY]
                model_name = f"{model_initailization_config[MODULE_KEY]}.{model_initailization_config[CLASS_KEY]}"

                model_initailization_config = InitializedModelDetail(model_serial_number=model_serial_number,
                                                                     model=model,
                                                                     param_grid_search=param_grid_search,
                                                                     model_name=model_name
                                                                     )
                initialized_model_list.append(model_initailization_config)
            
            #upadating class property 
            self.initialized_model_list = initialized_model_list

            logging.info(f"The initialised model details are: [{initialized_model_list}]")

            return self.initialized_model_list
        
        except Exception as e:
             raise HousingException(e,sys) from e
        


    def execute_grid_search_operation(self,initialized_model:InitializedModelDetail,
                                      input_feature,output_feature)-> GridSearchedBestModel:

        """
        excute_grid_search_operation(): function will perform paramter search operation and
        it will return you the best optimistic  model with best paramter:
        estimator: Model object
        param_grid: dictionary of paramter to perform search operation
        input_feature: your all input features
        output_feature: Target/Dependent features
        ================================================================================
        return: Function will return GridSearchOperation object
        """
        try:
            # instantiating GridSearchCV class
            
           
            grid_search_cv_ref = ModelFactory.get_class_and_module(module_name=self.grid_search_cv_module,
                                                             class_name=self.grid_search_class_name
                                                             )
            
            logging.info(f"Setting estimator and parameters to search with GridSearchCv classs")

            grid_search_cv = grid_search_cv_ref(estimator=initialized_model.model,
                                                param_grid=initialized_model.param_grid_search)
            grid_search_cv = ModelFactory.update_property_of_class(grid_search_cv,
                                                                   self.grid_search_property_data)             
            message = f'{">>"* 30} f"Training {type(initialized_model.model).__name__} Started." {"<<"*30}'
            logging.info(message)
            grid_search_cv.fit(input_feature, output_feature)



            grid_searched_best_model = GridSearchedBestModel(model_serial_number=initialized_model.model_serial_number,
                                                             model=initialized_model.model,
                                                             best_model=grid_search_cv.best_estimator_,
                                                             best_parameters= grid_search_cv.best_params_,
                                                             best_score=grid_search_cv.best_score_
                                                             )
            
            logging.info(f"Best model details for {initialized_model.model} are {grid_searched_best_model}")

            return grid_searched_best_model

        except Exception as e:
             raise HousingException(e,sys) from e
        

    def perform_best_parameter_search_for_initialized_model(self, initialized_model: InitializedModelDetail,
                                                             input_feature,
                                                             output_feature) -> GridSearchedBestModel:
        """
        This function will perform paramter search operation and
        it will return you the best optimistic  model with best paramter:
        estimator: Model object
        param_grid: dictionary of paramter to perform search operation
        input_feature: your all input features
        output_feature: Target/Dependent features
        ================================================================================
        return: Function will return a GridSearchOperation
        """
        try:
            return self.execute_grid_search_operation(initialized_model=initialized_model,
                                                      input_feature=input_feature,
                                                      output_feature=output_feature)
        except Exception as e:
            raise HousingException(e, sys) from e
        

    def perform_best_parameter_search_for_initialized_models(self,
                                                             initialized_model_list:List[InitializedModelDetail],
                                                             input_feature,
                                                             output_feature)->List[GridSearchedBestModel]:
        """
        This function will perfom GridSearchCv Operation for each model in list of InitializedModelDetail
        and return the list of best models found after performing GridSearchCv
        """
        try:
            self.grid_searched_best_model_list =[]
            for initialized_model_list in initialized_model_list:
                  grid_searched_best_model = self.perform_best_parameter_search_for_initialized_model(
                       initialized_model=initialized_model_list,
                       input_feature=input_feature,
                       output_feature=output_feature)

                  logging.info(f"adding {grid_searched_best_model} into grid_searched_best_model_list")
                  self.grid_searched_best_model_list.append(grid_searched_best_model)

            return self.grid_searched_best_model_list

        except Exception as e:
             raise HousingException(e,sys) from e
        

    @staticmethod
    def get_model_detail(model_details: List[InitializedModelDetail],
                         model_serial_number: str) -> InitializedModelDetail:
        """
        This function return ModelDetail
        """
        try:
            for model_data in model_details:
                if model_data.model_serial_number == model_serial_number:
                    return model_data
        except Exception as e:
            raise HousingException(e, sys) from e
        


    @staticmethod
    def get_best_model_from_grid_searched_best_model_list(grid_searched_best_model_list:List[GridSearchedBestModel],
                                                          base_accuracy =0.6
                                                          ):
        """
        This function will return best model based on trainig dataset with greater accuracy 
        """
        try:
            best_model = None
            for grid_searched_best_model in grid_searched_best_model_list:
                  if base_accuracy <= grid_searched_best_model.best_score:
                       logging.info(f"Acceptable model found: {grid_searched_best_model}")
                       base_accuracy = grid_searched_best_model.best_score

                       best_model = grid_searched_best_model
            
            if not best_model:
                 raise Exception(f"None of Model has base accuracy: {base_accuracy}")
            
            logging.info(f"Best Model: {best_model}")

            return best_model
        
        except Exception as e:
             raise HousingException(e,sys) from e
        

    def get_best_model(self,X,y,base_accuracy=0.6) -> BestModel:
        try:
             logging.info("Started Initializing model from config file")
             initialized_model_list = self.get_initialised_model_list()
             logging.info(f"Initialized model: {initialized_model_list}")
             grid_searched_best_model_list = self.perform_best_parameter_search_for_initialized_models(
                initialized_model_list=initialized_model_list,
                input_feature=X,
                output_feature=y
             )

             return ModelFactory.get_best_model_from_grid_searched_best_model_list(grid_searched_best_model_list,
                                                                                  base_accuracy=base_accuracy)
        
        except Exception as e:
             raise HousingException(e,sys) from e
        


        
