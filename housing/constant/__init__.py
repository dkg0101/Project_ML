import os,sys
from datetime import datetime


ROOT_DIR = os.getcwd()

CONFIG_DIR_NAME = "config"
CONFIG_FILE_NAME = "config.yaml"
CONFIG_FILE_PATH ="D:\Project_ML\config\cofig.yaml" #os.path.join(ROOT_DIR,CONFIG_DIR_NAME,CONFIG_FILE_NAME)

CURRENT_TIME_STAMP = f"{datetime.now().strftime('%y-%m-%d-%H-%M-%S')}"



#Trainig pipeline related variable 
TRAINING_PIPELINE_CONFIG_KEY = 'training_pipeline_config'
TRAINING_PIPELINE_ARTIFACT_DIR_KEY = 'artifact_dir'
TRAINING_PIPELINE_NAME_KEY = 'pipeline_name'
