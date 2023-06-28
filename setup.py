from setuptools import setup,find_packages
from typing import List 

PROJECT_NAME='housing-predictor'
VERSION ='0.0.3'
AUTHOR = 'Dhananjay Gurav'
DESCRIPTION = 'First end to end ml project with deployments'
PACKAGES = ["housing"]
REQUIREMENT_FILE_NAME = "requirements.txt"


def get_requirements_list():
    """
    Description: This function is going to return list of requirement
    mention in requirements.txt file 
    return this function is going to return a list which  contain name 
    of libraries mentioned in requirements.txt file
    """
    
    with open (REQUIREMENT_FILE_NAME,"r") as requirement_file:
        return requirement_file.readlines().remove("-e .")

    




setup(
    name=PROJECT_NAME,
    version= VERSION,
    author = AUTHOR,
    description=DESCRIPTION,
    packages=  find_packages() ,
    install_requires= get_requirements_list()
)



if __name__ == "__main__":
    requirements_list = get_requirements_list()

  