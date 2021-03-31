
# SageMaker-PyTorch-Recommender  :bar_chart: :chart_with_upwards_trend:


This is a project to create a simple recommender system using PyTorch and training it on AWS SageMaker. We use some data with Amazon product ratings and train a standard matrix factorization algorithm.


Hope someone will find it useful.


To run the full pipeline first run the script ``create_iam_role.py`` to give SageMaker the relevant access.
```
python3 create_iam_role.py
```

Then select a unique S3 bucket name in `pipeline.cfg` and run `main.py`:

```
python3 main.py
```

The pipeline carries out the following two steps:

1. Runs the ``data_preprocessing.ipynb`` jupyter notebook, which downloads the data and puts it in an S3 bucket
2. Trains the model ``recommender.py`` on a SageMaker instance with a PyTorch image


You will need to store your AWS access key e.g. in a ``~/.aws/credentials`` file and specify the AWS region e.g. in ``~/.aws/config``.


Tested on Ubuntu 20.04.


## Requirements

* pandas
* jupyter
* scikit-learn
* boto3
* sagemaker
* torch


## SageMaker

AWS SageMaker is a complete solution for ML projects, including data labelling (Ground Truth), graphical data preprocessing (DataWrangler), provisioning of resources for model training (Training Jobs), online Jupyter notebooks (SageMaker Studio and SageMaker Notebooks), model deployment, experiments tracking, and more.


### SageMaker Studio and Notebooks

A Sagemaker notebook is a Jupyter notebook running on AWS provisioned resources, and Sagemaker Studio is a JupyterLab instance, which is a wrapper around notebooks with additional functionality.

You can launch SageMaker studio/notebooks in the AWS console as described at: https://docs.aws.amazon.com/sagemaker/latest/dg/gs-setup-working-env.html.


## Future functionality

* Sagemaker model deployment
* Speed-up of training
* Keep track of training runs with SageMaker Experiments
* Create a Deep Recommender System
* Unit tests

