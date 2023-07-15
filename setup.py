from setuptools import setup,find_packages
from typing import List 

PROJECT_NAME='housing-predictor'
VERSION ='0.0.3'
AUTHOR = 'Dhananjay Gurav'
DESCRIPTION = 'First end to end ml project with deployments'
PACKAGES = ["housing"]
REQUIREMENT_FILE_NAME = "requirements.txt"
HYPEN_E_DOT = '-e .'

def get_requirements_list()->List[str]:
    """
    Description: This function is going to return list of requirement
    mention in requirements.txt file 
    return this function is going to return a list which  contain name 
    of libraries mentioned in requirements.txt file
    """
    requirements=[]
    with open(REQUIREMENT_FILE_NAME) as requirement_file:
        requirements = requirement_file.readlines()
        requirements=[req.replace('\n','') for req in requirements]
        
        if HYPEN_E_DOT in requirements:
                requirements.remove(HYPEN_E_DOT)


    return requirements



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

  