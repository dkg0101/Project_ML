from housing.pipeline.pipeline import Pipeline
from housing.exception import HousingException
from housing.logger import logging
from housing.config.configuration import Configuration
from housing.componant.data_transformation import DataTransformation
import os


def main():
    try:
        pipeline = Pipeline()
        # pipeline.run_pipeline()
        pipeline.start()
        logging.info("main function execution completed.")
       
    except Exception as e:
        logging.error(f"{e}")
        print(e)



if __name__=="__main__":
    main()