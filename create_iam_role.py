
import sys
import json
from time import sleep

import boto3


def create_sagemaker_role_if_not_exists(role_name: str) -> int:
    """
    Create a role for SageMaker to access S3 and run jobs.

    :param name: Name of the role

    :return: Return code, 0 if success

    """
    iam = boto3.resource('iam')

    if role_name not in [r.name for r in iam.roles.iterator()]:

        assumed_policy = {
                             "Version": "2012-10-17",
                             "Statement": [
                                 {
                                     "Effect": "Allow",
                                     "Principal": {
                                         "Service": ["sagemaker.amazonaws.com"]
                                     },
                                     "Action": ["sts:AssumeRole"]
                                 }
                              ]
                          }

        assumed_policy_json = json.dumps(assumed_policy)

        role = iam.create_role(RoleName                 = role_name,
                               AssumeRolePolicyDocument = assumed_policy_json)

        sagemaker_policy = 'arn:aws:iam::aws:policy/AmazonSageMakerFullAccess'
        s3_policy = 'arn:aws:iam::aws:policy/AmazonS3FullAccess'
        role.attach_policy(PolicyArn=sagemaker_policy)
        role.attach_policy(PolicyArn=s3_policy)

    return 0


if __name__ == '__main__':

    sys.exit(create_sagemaker_role_if_not_exists('SageMakerRecSys'))
    sleep(5) # Wait for the role to be created

