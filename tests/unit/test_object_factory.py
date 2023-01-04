from __future__ import absolute_import

import pytest
from mock import MagicMock, Mock
from sagemaker.sklearn.processing import SKLearnProcessor

from smtoolkit.object_factory import sm_processing
from smtoolkit.sklearn.processing import SKLearnProcessorBuilder

BUCKET_NAME = "mybucket"
REGION = "us-west-2"
ROLE = "arn:aws:iam::012345678901:role/SageMakerRole"
MOCKED_S3_URI = "s3://mocked_s3_uri_from_upload_data"
ECR_HOSTNAME = "ecr.us-west-2.amazonaws.com"
CUSTOM_IMAGE_URI = "012345678901.dkr.ecr.us-west-2.amazonaws.com/my-custom-image-uri"


@pytest.fixture()
def sagemaker_session():
    boto_mock = Mock(name="boto_session", region_name=REGION)
    session_mock = MagicMock(
        name="sagemaker_session",
        boto_session=boto_mock,
        boto_region_name=REGION,
        config=None,
        local_mode=False,
    )
    session_mock.default_bucket = Mock(name="default_bucket", return_value=BUCKET_NAME)

    session_mock.upload_data = Mock(name="upload_data", return_value=MOCKED_S3_URI)
    session_mock.download_data = Mock(name="download_data")
    session_mock.expand_role.return_value = ROLE
    session_mock.describe_processing_job = MagicMock(
        name="describe_processing_job",
        return_value=_get_describe_response_inputs_and_ouputs(),
    )
    return session_mock


def test_sm_processing_sklearn(sklearn_version, sagemaker_session):
    params = {
        "role": ROLE,
        "instance_type": "ml.m4.xlarge",
        "framework_version": sklearn_version,
        "instance_count": 1,
        "sagemaker_session": sagemaker_session,
    }
    processor = sm_processing.get_or_create(processor="SKLearnProcessor", **params)

    assert isinstance(processor, SKLearnProcessor)


def test_sm_processing_sklearn_exception():
    with pytest.raises(Exception):
        assert sm_processing.get_or_create(processor="ProcessorFaker")


def test_sm_processing_processors():
    processors = sm_processing.builders
    skLearn_processor = processors["SKLearnProcessor"]

    assert isinstance(processors, dict)
    assert isinstance(skLearn_processor, SKLearnProcessorBuilder)


def _get_describe_response_inputs_and_ouputs():
    return {
        "ProcessingInputs": _get_expected_args_all_parameters(None)["inputs"],
        "ProcessingOutputConfig": _get_expected_args_all_parameters(None)[
            "output_config"
        ],
    }


def _get_expected_args_all_parameters(job_name):
    return {
        "inputs": [
            {
                "InputName": "my_dataset",
                "AppManaged": False,
                "S3Input": {
                    "S3Uri": "s3://path/to/my/dataset/census.csv",
                    "LocalPath": "/container/path/",
                    "S3DataType": "S3Prefix",
                    "S3InputMode": "File",
                    "S3DataDistributionType": "FullyReplicated",
                    "S3CompressionType": "None",
                },
            },
            {
                "InputName": "s3_input",
                "AppManaged": False,
                "S3Input": {
                    "S3Uri": "s3://path/to/my/dataset/census.csv",
                    "LocalPath": "/container/path/",
                    "S3DataType": "S3Prefix",
                    "S3InputMode": "File",
                    "S3DataDistributionType": "FullyReplicated",
                    "S3CompressionType": "None",
                },
            },
            {
                "InputName": "redshift_dataset_definition",
                "AppManaged": True,
                "DatasetDefinition": {
                    "DataDistributionType": "FullyReplicated",
                    "InputMode": "File",
                    "LocalPath": "/opt/ml/processing/input/dd",
                    "RedshiftDatasetDefinition": {
                        "ClusterId": "cluster_id",
                        "Database": "database",
                        "DbUser": "db_user",
                        "QueryString": "query_string",
                        "ClusterRoleArn": "cluster_role_arn",
                        "OutputS3Uri": "output_s3_uri",
                        "KmsKeyId": "kms_key_id",
                        "OutputFormat": "CSV",
                        "OutputCompression": "SNAPPY",
                    },
                },
            },
            {
                "InputName": "athena_dataset_definition",
                "AppManaged": True,
                "DatasetDefinition": {
                    "DataDistributionType": "FullyReplicated",
                    "InputMode": "File",
                    "LocalPath": "/opt/ml/processing/input/dd",
                    "AthenaDatasetDefinition": {
                        "Catalog": "catalog",
                        "Database": "database",
                        "QueryString": "query_string",
                        "OutputS3Uri": "output_s3_uri",
                        "WorkGroup": "workgroup",
                        "KmsKeyId": "kms_key_id",
                        "OutputFormat": "AVRO",
                        "OutputCompression": "ZLIB",
                    },
                },
            },
            {
                "InputName": "code",
                "AppManaged": False,
                "S3Input": {
                    "S3Uri": MOCKED_S3_URI,
                    "LocalPath": "/opt/ml/processing/input/code",
                    "S3DataType": "S3Prefix",
                    "S3InputMode": "File",
                    "S3DataDistributionType": "FullyReplicated",
                    "S3CompressionType": "None",
                },
            },
        ],
        "output_config": {
            "Outputs": [
                {
                    "OutputName": "my_output",
                    "AppManaged": False,
                    "S3Output": {
                        "S3Uri": "s3://uri/",
                        "LocalPath": "/container/path/",
                        "S3UploadMode": "EndOfJob",
                    },
                },
                {
                    "OutputName": "feature_store_output",
                    "AppManaged": True,
                    "FeatureStoreOutput": {"FeatureGroupName": "FeatureGroupName"},
                },
            ],
            "KmsKeyId": "arn:aws:kms:us-west-2:012345678901:key/output-kms-key",
        },
        "job_name": job_name,
        "resources": {
            "ClusterConfig": {
                "InstanceType": "ml.m4.xlarge",
                "InstanceCount": 1,
                "VolumeSizeInGB": 100,
                "VolumeKmsKeyId": "arn:aws:kms:us-west-2:012345678901:key/volume-kms-key",
            }
        },
        "stopping_condition": {"MaxRuntimeInSeconds": 3600},
        "app_specification": {
            "ImageUri": "012345678901.dkr.ecr.us-west-2.amazonaws.com/my-custom-image-uri",
            "ContainerArguments": ["--drop-columns", "'SelfEmployed'"],
            "ContainerEntrypoint": [
                "python3",
                "/opt/ml/processing/input/code/processing_code.py",
            ],
        },
        "environment": {"my_env_variable": "my_env_variable_value"},
        "network_config": {
            "EnableNetworkIsolation": True,
            "EnableInterContainerTrafficEncryption": True,
            "VpcConfig": {
                "SecurityGroupIds": ["my_security_group_id"],
                "Subnets": ["my_subnet_id"],
            },
        },
        "role_arn": ROLE,
        "tags": [{"Key": "my-tag", "Value": "my-tag-value"}],
        "experiment_config": {"ExperimentName": "AnExperiment"},
    }
