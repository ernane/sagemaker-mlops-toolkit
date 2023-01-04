from __future__ import absolute_import

from pathlib import Path

import pytest
from mock import MagicMock, Mock, patch
from packaging import version
from sagemaker.dataset_definition.inputs import (
    AthenaDatasetDefinition,
    DatasetDefinition,
    RedshiftDatasetDefinition,
    S3Input,
)
from sagemaker.fw_utils import UploadedCode
from sagemaker.network import NetworkConfig
from sagemaker.processing import FeatureStoreOutput, ProcessingInput, ProcessingOutput

from smtoolkit.sklearn.processing import SKLearnProcessorBuilder
from smtoolkit.xgboost.processing import XGBoostProcessorBuilder

BUCKET_NAME = "mybucket"
REGION = "us-west-2"
ROLE = "arn:aws:iam::012345678901:role/SageMakerRole"
MOCKED_S3_URI = "s3://mocked_s3_uri_from_upload_data"
ECR_HOSTNAME = "ecr.us-west-2.amazonaws.com"
CUSTOM_IMAGE_URI = "012345678901.dkr.ecr.us-west-2.amazonaws.com/my-custom-image-uri"


@pytest.fixture()
def uploaded_code(
    s3_prefix="s3://mocked_s3_uri_from_upload_data/my_job_name/source/sourcedir.tar.gz",
    script_name="processing_code.py",
):
    return UploadedCode(s3_prefix=s3_prefix, script_name=script_name)


@pytest.fixture()
def skLearn_processor():
    return SKLearnProcessorBuilder()


@pytest.fixture()
def xgboost_processor():
    return XGBoostProcessorBuilder()


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
    exists_mock,
    isfile_mock,
    botocore_resolver,
    sagemaker_session,
    sklearn_version,
    skLearn_processor,
):
    botocore_resolver.return_value.construct_endpoint.return_value = {
        "hostname": ECR_HOSTNAME
    }

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


@patch("sagemaker.utils._botocore_resolver")
@patch("os.path.exists", return_value=True)
@patch("os.path.isfile", return_value=True)
def test_sklearn_with_all_parameters(
    exists_mock,
    isfile_mock,
    botocore_resolver,
    sklearn_version,
    sagemaker_session,
    skLearn_processor,
):
    botocore_resolver.return_value.construct_endpoint.return_value = {
        "hostname": ECR_HOSTNAME
    }

    processor = skLearn_processor(
        role=ROLE,
        framework_version=sklearn_version,
        instance_type="ml.m4.xlarge",
        instance_count=1,
        volume_size_in_gb=100,
        volume_kms_key="arn:aws:kms:us-west-2:012345678901:key/volume-kms-key",
        output_kms_key="arn:aws:kms:us-west-2:012345678901:key/output-kms-key",
        max_runtime_in_seconds=3600,
        base_job_name="my_sklearn_processor",
        env={"my_env_variable": "my_env_variable_value"},
        tags=[{"Key": "my-tag", "Value": "my-tag-value"}],
        network_config=NetworkConfig(
            subnets=["my_subnet_id"],
            security_group_ids=["my_security_group_id"],
            enable_network_isolation=True,
            encrypt_inter_container_traffic=True,
        ),
        sagemaker_session=sagemaker_session,
    )

    processor.run(
        code="/local/path/to/processing_code.py",
        inputs=_get_data_inputs_all_parameters(),
        outputs=_get_data_outputs_all_parameters(),
        arguments=["--drop-columns", "'SelfEmployed'"],
        wait=True,
        logs=False,
        job_name="my_job_name",
        experiment_config={"ExperimentName": "AnExperiment"},
    )

    expected_args = _get_expected_args_all_parameters(processor._current_job_name)
    sklearn_image_uri = (
        "246618743249.dkr.ecr.us-west-2.amazonaws.com/sagemaker-scikit-learn:{}-cpu-py3"
    ).format(sklearn_version)
    expected_args["app_specification"]["ImageUri"] = sklearn_image_uri

    sagemaker_session.process.assert_called_with(**expected_args)


@patch("sagemaker.utils._botocore_resolver")
@patch("os.path.exists", return_value=True)
@patch("os.path.isfile", return_value=True)
def test_xgboost_processor_with_required_parameters(
    exists_mock,
    isfile_mock,
    botocore_resolver,
    sagemaker_session,
    xgboost_framework_version,
    xgboost_processor,
):
    botocore_resolver.return_value.construct_endpoint.return_value = {
        "hostname": ECR_HOSTNAME
    }

    processor = xgboost_processor(
        role=ROLE,
        instance_type="ml.m4.xlarge",
        framework_version=xgboost_framework_version,
        instance_count=1,
        sagemaker_session=sagemaker_session,
    )

    code = Path("/tmp/processing_code.py")
    code.touch(exist_ok=True)

    processor.run(code="/tmp/processing_code.py")

    expected_args = _get_expected_args_modular_code(processor._current_job_name)

    if version.parse(xgboost_framework_version) < version.parse("1.2-1"):
        xgboost_image_uri = (
            "246618743249.dkr.ecr.us-west-2.amazonaws.com/sagemaker-xgboost:{}-cpu-py3"
        ).format(xgboost_framework_version)
    else:
        xgboost_image_uri = (
            "246618743249.dkr.ecr.us-west-2.amazonaws.com/sagemaker-xgboost:{}"
        ).format(xgboost_framework_version)

    expected_args["app_specification"]["ImageUri"] = xgboost_image_uri

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


def _get_data_inputs_all_parameters():
    return [
        ProcessingInput(
            source="s3://path/to/my/dataset/census.csv",
            destination="/container/path/",
            input_name="my_dataset",
            s3_data_type="S3Prefix",
            s3_input_mode="File",
            s3_data_distribution_type="FullyReplicated",
            s3_compression_type="None",
        ),
        ProcessingInput(
            input_name="s3_input",
            s3_input=S3Input(
                s3_uri="s3://path/to/my/dataset/census.csv",
                local_path="/container/path/",
                s3_data_type="S3Prefix",
                s3_input_mode="File",
                s3_data_distribution_type="FullyReplicated",
                s3_compression_type="None",
            ),
        ),
        ProcessingInput(
            input_name="redshift_dataset_definition",
            app_managed=True,
            dataset_definition=DatasetDefinition(
                data_distribution_type="FullyReplicated",
                input_mode="File",
                local_path="/opt/ml/processing/input/dd",
                redshift_dataset_definition=RedshiftDatasetDefinition(
                    cluster_id="cluster_id",
                    database="database",
                    db_user="db_user",
                    query_string="query_string",
                    cluster_role_arn="cluster_role_arn",
                    output_s3_uri="output_s3_uri",
                    kms_key_id="kms_key_id",
                    output_format="CSV",
                    output_compression="SNAPPY",
                ),
            ),
        ),
        ProcessingInput(
            input_name="athena_dataset_definition",
            app_managed=True,
            dataset_definition=DatasetDefinition(
                data_distribution_type="FullyReplicated",
                input_mode="File",
                local_path="/opt/ml/processing/input/dd",
                athena_dataset_definition=AthenaDatasetDefinition(
                    catalog="catalog",
                    database="database",
                    query_string="query_string",
                    output_s3_uri="output_s3_uri",
                    work_group="workgroup",
                    kms_key_id="kms_key_id",
                    output_format="AVRO",
                    output_compression="ZLIB",
                ),
            ),
        ),
    ]


def _get_data_outputs_all_parameters():
    return [
        ProcessingOutput(
            source="/container/path/",
            destination="s3://uri/",
            output_name="my_output",
            s3_upload_mode="EndOfJob",
        ),
        ProcessingOutput(
            output_name="feature_store_output",
            app_managed=True,
            feature_store_output=FeatureStoreOutput(
                feature_group_name="FeatureGroupName"
            ),
        ),
    ]


def _get_expected_args_modular_code(job_name, code_s3_uri=f"s3://{BUCKET_NAME}"):
    return {
        "inputs": [
            {
                "InputName": "code",
                "AppManaged": False,
                "S3Input": {
                    "S3Uri": f"{code_s3_uri}/{job_name}/source/sourcedir.tar.gz",
                    "LocalPath": "/opt/ml/processing/input/code/",
                    "S3DataType": "S3Prefix",
                    "S3InputMode": "File",
                    "S3DataDistributionType": "FullyReplicated",
                    "S3CompressionType": "None",
                },
            },
            {
                "InputName": "entrypoint",
                "AppManaged": False,
                "S3Input": {
                    "S3Uri": f"{code_s3_uri}/{job_name}/source/runproc.sh",
                    "LocalPath": "/opt/ml/processing/input/entrypoint",
                    "S3DataType": "S3Prefix",
                    "S3InputMode": "File",
                    "S3DataDistributionType": "FullyReplicated",
                    "S3CompressionType": "None",
                },
            },
        ],
        "output_config": {"Outputs": []},
        "experiment_config": None,
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
                "/bin/bash",
                "/opt/ml/processing/input/entrypoint/runproc.sh",
            ],
        },
        "environment": None,
        "network_config": None,
        "role_arn": ROLE,
        "tags": None,
        "experiment_config": None,
    }
