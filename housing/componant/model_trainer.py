import os,sys
from typing import List

from housing.logger import logging 
from housing.exception import HousingException
from housing.entity.model_factory import *
from housing.entity.config_entity import ModelTrainerConfig
from housing.entity.artifact_entity import DataTransformationArtifact,ModelTrainerArtifact
from housing.util.util import load_numpy_array_data,save_object,load_object

from housing.entity.model_factory import evaluate_regression_model



class HousingEstimatorModel:
    def __init__(self, preprocessing_object, trained_model_object):
        """
        TrainedModel constructor
        preprocessing_object: preprocessing_object
        trained_model_object: trained_model_object

        """
        self.preprocessing_object = preprocessing_object
        self.trained_model_object = trained_model_object

    def predict(self,X):
        """
        Fuction that performs preprocessing on input data and 
        after that makes prediction on transformed features
        """

        transformed_feature  = self.preprocessing_object.transform(X)

        return self.trained_model_object.predict(transformed_feature)
    
    def __repr__(self) :
        return f"{type(self.trained_model_object).__name__}()"
    
    def __str__(self) :
        return f"{type(self.trained_model_object).__name__}()"
    


class ModelTrainer:
    
    def __init__(self,model_trainer_config:ModelTrainerConfig,data_transformation_artifact:DataTransformationArtifact):
        try:
            logging.info("*******Model Training log started************")
            self.model_trainer_config = model_trainer_config
            self.data_transformation_artifact = data_transformation_artifact



        except Exception as e:
            raise HousingException(e,sys) from e
        

    def intiate_model_trainer(self)-> ModelTrainerArtifact:
        try:
             
            logging.info(f"Loading transformed training dataset")
            transformed_train_file_path = self.data_transformation_artifact.transformed_train_file_path
            train_array = load_numpy_array_data(file_path=transformed_train_file_path)

            logging.info(f"Loading transformed testing dataset")
            transformed_test_file_path = self.data_transformation_artifact.transformed_test_file_path
            test_array = load_numpy_array_data(file_path=transformed_test_file_path)
            
            logging.info(f"Splitting training and testing input and target feature")
            x_train,y_train,x_test,y_test = train_array[:,:-1],train_array[:,-1],test_array[:,:-1],test_array[:,-1]
            
            logging.info(f"Extracting model config file path")
            model_config_file_path = self.model_trainer_config.model_config_file_path

            logging.info(f"Initializing Model factory class using model config file;{model_config_file_path}")
            model_factory = ModelFactory(model_config_path=model_config_file_path)


            base_accuracy = self.model_trainer_config.base_accuracy
            logging.info(f"Expected Accuracy : {base_accuracy}")

            logging.info("Initiating operation Model Selection")
            best_model = model_factory.get_best_model(X=x_train,y=y_train,base_accuracy=base_accuracy)
        
            logging.info(f'Best Model found on training dataset is {best_model}')

            logging.info("Getting trained model list")
            grid_searched_best_model_list:List[GridSearchedBestModel] =model_factory.grid_searched_best_model_list

            model_list = [model.best_model for model in grid_searched_best_model_list]
            logging.info(f"Evaluation all trained model on training and testing dataset both")
            metric_info_artifact:MetricInfoArtifact = evaluate_regression_model(model_list=model_list,X_train=x_train,X_test=x_test,y_train=y_train,y_test=y_test,base_accuracy=base_accuracy)
            
            logging.info(f"Best Model found on both training and testing dataset")

            preprocessing_obj = load_object(file_path=self.data_transformation_artifact.preprocessed_object_file_path)
            model_obj = metric_info_artifact.model_object

            trained_model_file_path = self.model_trainer_config.trained_model_file_path
            #To save final model by combining both preprocessed and trained best model
            housing_model = HousingEstimatorModel(preprocessing_object=preprocessing_obj,trained_model_object=model_obj)
            logging.info(f"Saving model at path: {trained_model_file_path}")
            save_object(file_path=trained_model_file_path,obj=housing_model)

            model_trainer_artifact =  ModelTrainerArtifact(is_trained=True,message="MOdel trained successfully",
            trained_model_file_path=trained_model_file_path,
            train_rmse=metric_info_artifact.train_rmse,
            test_rmse=metric_info_artifact.test_rmse,
            train_accuracy=metric_info_artifact.train_accuracy,
            test_accuracy=metric_info_artifact.test_accuracy,
            model_accuracy=metric_info_artifact.model_accuracy
            )



            logging.info(f"model_trainer_artifact:[{model_trainer_artifact}]")
            return model_trainer_artifact



        except Exception as e:
            raise HousingException(e,sys) from e
        








