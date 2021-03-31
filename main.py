
"""
PyTorch Recommender in SageMaker

"""

# Built-in packages
import sys

# External packages
from nbformat import read as read_notebook
from nbconvert.preprocessors import ExecutePreprocessor
from sagemaker.pytorch import PyTorch

# This package
from config import config


def main() -> int:

    print("Running the data preprocessing notebook.")

    with open('data_preprocessing.ipynb') as f:
        notebook = read_notebook(f, as_version=4)
    ExecutePreprocessor(timeout=-1).preprocess(notebook) 


    print("Training PyTorch matrix factorization model.")

    estimator = PyTorch(entry_point       = 'recommender.py',
                        role              = 'SageMakerRecSys',
                        instance_type     = 'ml.m5.large',
                        instance_count    = 1,
                        py_version        = 'py36',
                        framework_version = '1.8')

    estimator.fit("s3://" + config['AWS']['S3_BUCKET'])

    return 0


if __name__ == '__main__':

    sys.exit(main())

