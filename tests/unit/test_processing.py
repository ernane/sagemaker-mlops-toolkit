from __future__ import absolute_import

import pytest
from mock import MagicMock, Mock, patch

from smtoolkit.sklearn.processing import SKLearnProcessorBuilder

BUCKET_NAME = "mybucket"
REGION = "us-west-2"
ROLE = "arn:aws:iam::012345678901:role/SageMakerRole"
MOCKED_S3_URI = "s3://mocked_s3_uri_from_upload_data"
ECR_HOSTNAME = "ecr.us-west-2.amazonaws.com"
CUSTOM_IMAGE_URI = "012345678901.dkr.ecr.us-west-2.amazonaws.com/my-custom-image-uri"


@pytest.fixture(scope="session")
def sklearn_version():
    return "1.0-1"


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


@patch("sagemaker.utils._botocore_resolver")
@patch("os.path.exists", return_value=True)
@patch("os.path.isfile", return_value=True)
def test_sklearn_processor_with_required_parameters(
    exists_mock, isfile_mock, botocore_resolver, sagemaker_session, sklearn_version
):
    botocore_resolver.return_value.construct_endpoint.return_value = {
        "hostname": ECR_HOSTNAME
    }
    skLearn_processor = SKLearnProcessorBuilder()

    processor = skLearn_processor(
        role=ROLE,
        instance_type="ml.m4.xlarge",
        framework_version=sklearn_version,
        instance_count=1,
        sagemaker_session=sagemaker_session,
    )

    processor.run(code="/local/path/to/processing_code.py")

    expected_args = _get_expected_args(processor._current_job_name)

    sklearn_image_uri = (
        "246618743249.dkr.ecr.us-west-2.amazonaws.com/sagemaker-scikit-learn:{}-cpu-py3"
    ).format(sklearn_version)
    expected_args["app_specification"]["ImageUri"] = sklearn_image_uri
    sagemaker_session.process.assert_called_with(**expected_args)


def _get_expected_args(job_name, code_s3_uri="s3://mocked_s3_uri_from_upload_data"):
    return {
        "inputs": [
            {
                "InputName": "code",
                "AppManaged": False,
                "S3Input": {
                    "S3Uri": code_s3_uri,
                    "LocalPath": "/opt/ml/processing/input/code",
                    "S3DataType": "S3Prefix",
                    "S3InputMode": "File",
                    "S3DataDistributionType": "FullyReplicated",
                    "S3CompressionType": "None",
                },
            },
        ],
        "output_config": {"Outputs": []},
        "job_name": job_name,
        "resources": {
            "ClusterConfig": {
                "InstanceType": "ml.m4.xlarge",
                "InstanceCount": 1,
                "VolumeSizeInGB": 30,
            }
        },
        "stopping_condition": None,
        "app_specification": {
            "ImageUri": CUSTOM_IMAGE_URI,
            "ContainerEntrypoint": [
                "python3",
                "/opt/ml/processing/input/code/processing_code.py",
            ],
        },
        "environment": None,
        "network_config": None,
        "role_arn": ROLE,
        "tags": None,
        "experiment_config": None,
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


def _get_describe_response_inputs_and_ouputs():
    return {
        "ProcessingInputs": _get_expected_args_all_parameters(None)["inputs"],
        "ProcessingOutputConfig": _get_expected_args_all_parameters(None)[
            "output_config"
        ],
    }
