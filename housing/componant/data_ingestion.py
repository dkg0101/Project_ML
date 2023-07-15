from housing.entity.config_entity import DataIngestionConfig
import sys,os
from housing.exception import HousingException
from housing.logger import logging
from housing.entity.artifact_entity import  DataIngestionArtifact
import numpy as np
from six.moves import urllib
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

class DataIngestion:

    def __init__(self,data_ingestion_config:DataIngestionConfig):
        try:
            logging.info(f"{'='*20}Data Ingestion log started.{'='*20} ")
            self.data_ingestion_config = data_ingestion_config

        except Exception as e:
            raise HousingException(e,sys)

    def download_housing_data(self):
        try:
            #To get remote url for download the dataset
            download_url= self.data_ingestion_config.dataset_download_url

            #Folder location to download file
            raw_data_dir = self.data_ingestion_config.raw_data_dir

            # To recreate folder
            if os.path.exists(raw_data_dir):
                os.remove(raw_data_dir)

            

            os.makedirs(raw_data_dir,exist_ok=True)
            logging.info("raw data folder created...")



            # To get filename
            housing_file_name = os.path.basename(download_url)

            #file path
            raw_data_file_path = os.path.join(raw_data_dir,housing_file_name)

            logging.info(f"Downloading file from: [{download_url}] into [{raw_data_file_path}]")
            urllib.request.urlretrieve(download_url,raw_data_file_path)
            logging.info(f"file {raw_data_file_path} has been downloaded succesfully!")

            return raw_data_file_path
        
        except Exception as e:
            raise HousingException(e,sys) from e



        

    def split_data_as_train_test(self):
        try:
            raw_data_dir = self.data_ingestion_config.raw_data_dir

            file_name = os.listdir(raw_data_dir)[0]

            housing_file_path = "F://MachineLearningProject//Project_ML//housing.csv" #os.path.join(raw_data_dir,file_name)

            logging.info(f"reading csv file; {housing_file_path}")

            housing_data_frame =pd.read_csv(housing_file_path)

            #defining target column

            housing_data_frame["income_cat"] = pd.cut(
                housing_data_frame["median_income"],
                bins=[0.0,1.5,3.0,4.5,6.0,np.inf],
                labels=[1,2,3,4,5]
            )

            logging.info('spliting data into train and test set')



            strat_train_set = None
            strat_test_set = None

            split = StratifiedShuffleSplit(n_splits=1,test_size=0.2,random_state=28)

            for train_index,test_index in split.split(housing_data_frame,housing_data_frame['income_cat']):
                strat_train_set = housing_data_frame.loc[train_index].drop(['income_cat'],axis =1)
                strat_test_set = housing_data_frame.loc[test_index].drop(['income_cat'],axis =1)

            train_file_path = os.path.join(self.data_ingestion_config.ingested_train_dir,file_name)

            test_file_path= os.path.join(self.data_ingestion_config.ingested_test_dir,file_name)


            if strat_train_set is not None:
                os.makedirs(self.data_ingestion_config.ingested_train_dir,exist_ok=True)
                logging.info(f"Exporting training data to file:[{train_file_path}]")
                strat_train_set.to_csv(train_file_path,index=False)



            if strat_test_set is not None:
                os.makedirs(self.data_ingestion_config.ingested_test_dir,exist_ok=True)

                logging.info(f"Exporting test data to file:[{test_file_path}]")

                strat_test_set.to_csv(test_file_path,index=False)

            data_ingestion_artifact = DataIngestionArtifact(train_file_path=train_file_path,
                                                        test_file_path=test_file_path,
                                                        is_ingested=True,
                                                        message="Data ingestion Completed...")       
        
        
            logging.info(f"data ingestion artifact:{data_ingestion_artifact}")

            return data_ingestion_artifact


        except Exception as e:
            raise HousingException(e,sys) from e

    def initiate_data_ingestion(self)->DataIngestionArtifact:
        try:
            raw_data_file_path =self.download_housing_data()
            return self.split_data_as_train_test()
        
        except Exception as e:
            raise HousingException(e,sys) from e
        
