from __future__ import absolute_import

import json

import boto3
import pytest
from packaging.version import Version
from sagemaker import image_uris

DEFAULT_REGION = "us-east-1"

HOSTING_NO_P2_REGIONS = [
    "af-south-1",
    "ap-east-1",
    "ca-central-1",
    "eu-central-1",
    "eu-north-1",
    "eu-west-2",
    "eu-west-3",
    "eu-south-1",
    "sa-east-1",
    "us-west-1",
]

TRAINING_NO_P2_REGIONS = [
    "af-south-1",
    "ap-east-1",
    "ap-southeast-1",
    "ap-southeast-2",
    "ca-central-1",
    "eu-central-1",
    "eu-north-1",
    "eu-west-2",
    "eu-west-3",
    "eu-south-1",
    "me-south-1",
    "sa-east-1",
    "us-west-1",
]

NO_M4_REGIONS = [
    "eu-west-3",
    "eu-north-1",
    "ap-east-1",
    "ap-northeast-1",  # it has m4.xl, but not enough in all AZs
    "sa-east-1",
    "me-south-1",
]

HOSTING_NO_P3_REGIONS = [
    "af-south-1",
    "ap-east-1",
    "ap-south-1",
    "ap-southeast-1",
    "ap-southeast-2",
    "ca-central-1",
    "eu-central-1",
    "eu-north-1",
    "eu-west-2",
    "eu-west-3",
    "eu-south-1",
    "sa-east-1",
    "us-west-1",
]

TRAINING_NO_P3_REGIONS = [
    "af-south-1",
    "ap-east-1",
    "ap-southeast-1",  # it has p3, but not enough
    "ap-southeast-2",  # it has p3, but not enough
    "ca-central-1",  # it has p3, but not enough
    "eu-central-1",  # it has p3, but not enough
    "eu-north-1",
    "eu-west-1",  # it has p3, but not enough
    "eu-west-2",  # it has p3, but not enough
    "eu-west-3",
    "eu-south-1",
    "me-south-1",
    "sa-east-1",
    "us-west-1",
    "ap-northeast-1",  # it has p3, but not enough
    "ap-south-1",
    "ap-northeast-2",  # it has p3, but not enough
    "us-east-2",  # it has p3, but not enough
]

FRAMEWORKS_FOR_GENERATED_VERSION_FIXTURES = (
    "chainer",
    "coach_mxnet",
    "coach_tensorflow",
    "inferentia_mxnet",
    "inferentia_tensorflow",
    "inferentia_pytorch",
    "mxnet",
    "neo_mxnet",
    "neo_pytorch",
    "neo_tensorflow",
    "pytorch",
    "pytorch_training_compiler",
    "ray_pytorch",
    "ray_tensorflow",
    "sklearn",
    "tensorflow",
    "vw",
    "xgboost",
    "spark",
    "huggingface",
    "autogluon",
    "huggingface_training_compiler",
)


@pytest.fixture(scope="session")
def boto_session(request):
    config = request.config.getoption("--boto-config")
    if config:
        return boto3.Session(**json.loads(config))
    else:
        return boto3.Session(region_name=DEFAULT_REGION)


@pytest.fixture(scope="session")
def account(boto_session):
    return boto_session.client("sts").get_caller_identity()["Account"]


@pytest.fixture(scope="session")
def region(boto_session):
    return boto_session.region_name


def pytest_generate_tests(metafunc):
    if "instance_type" in metafunc.fixturenames:
        boto_config = metafunc.config.getoption("--boto-config")
        parsed_config = json.loads(boto_config) if boto_config else {}
        region = parsed_config.get("region_name", DEFAULT_REGION)
        cpu_instance_type = (
            "ml.m5.xlarge" if region in NO_M4_REGIONS else "ml.m4.xlarge"
        )

        params = [cpu_instance_type]
        if not (region in HOSTING_NO_P3_REGIONS or region in TRAINING_NO_P3_REGIONS):
            params.append("ml.p3.2xlarge")
        elif not (region in HOSTING_NO_P2_REGIONS or region in TRAINING_NO_P2_REGIONS):
            params.append("ml.p2.xlarge")

        metafunc.parametrize("instance_type", params, scope="session")

    _generate_all_framework_version_fixtures(metafunc)


def _generate_all_framework_version_fixtures(metafunc):
    for fw in FRAMEWORKS_FOR_GENERATED_VERSION_FIXTURES:
        config = image_uris.config_for_framework(fw.replace("_", "-"))
        if "scope" in config:
            _parametrize_framework_version_fixtures(metafunc, fw, config)
        else:
            for image_scope in config.keys():
                if fw in ("xgboost", "sklearn"):
                    _parametrize_framework_version_fixtures(
                        metafunc, fw, config[image_scope]
                    )
                    # XGB and SKLearn use the same configs for training,
                    # inference, and graviton_inference. Break after first
                    # iteration to avoid duplicate KeyError
                    break
                fixture_prefix = f"{fw}_{image_scope}" if image_scope not in fw else fw
                _parametrize_framework_version_fixtures(
                    metafunc, fixture_prefix, config[image_scope]
                )


def _parametrize_framework_version_fixtures(metafunc, fixture_prefix, config):
    fixture_name = "{}_version".format(fixture_prefix)
    if fixture_name in metafunc.fixturenames:
        versions = list(config["versions"].keys()) + list(
            config.get("version_aliases", {}).keys()
        )
        metafunc.parametrize(fixture_name, versions, scope="session")

    latest_version = sorted(config["versions"].keys(), key=lambda v: Version(v))[-1]

    fixture_name = "{}_latest_version".format(fixture_prefix)
    if fixture_name in metafunc.fixturenames:
        metafunc.parametrize(fixture_name, (latest_version,), scope="session")

    if "huggingface" in fixture_prefix:
        _generate_huggingface_base_fw_latest_versions(
            metafunc, fixture_prefix, latest_version, "pytorch"
        )
        _generate_huggingface_base_fw_latest_versions(
            metafunc, fixture_prefix, latest_version, "tensorflow"
        )

    fixture_name = "{}_latest_py_version".format(fixture_prefix)
    if fixture_name in metafunc.fixturenames:
        config = config["versions"]
        py_versions = config[latest_version].get(
            "py_versions", config[latest_version].keys()
        )
        if "repository" in py_versions or "registries" in py_versions:
            # Config did not specify `py_versions` and is not arranged by py_version. Assume py3
            metafunc.parametrize(fixture_name, ("py3",), scope="session")
        else:
            metafunc.parametrize(
                fixture_name, (sorted(py_versions)[-1],), scope="session"
            )


def _generate_huggingface_base_fw_latest_versions(
    metafunc, fixture_prefix, huggingface_version, base_fw
):
    versions = _huggingface_base_fm_version(
        huggingface_version, base_fw, fixture_prefix
    )
    fixture_name = f"{fixture_prefix}_{base_fw}_latest_version"

    if fixture_name in metafunc.fixturenames:
        metafunc.parametrize(fixture_name, versions, scope="session")


def _huggingface_base_fm_version(huggingface_version, base_fw, fixture_prefix):
    config_name = (
        "huggingface-training-compiler"
        if "training_compiler" in fixture_prefix
        else "huggingface"
    )
    config = image_uris.config_for_framework(config_name)
    if "training" in fixture_prefix:
        hf_config = config.get("training")
    else:
        hf_config = config.get("inference")
    original_version = huggingface_version
    if "version_aliases" in hf_config:
        huggingface_version = hf_config.get("version_aliases").get(
            huggingface_version, huggingface_version
        )
    version_config = hf_config.get("versions").get(huggingface_version)
    versions = list()

    for key in list(version_config.keys()):
        if key.startswith(base_fw):
            base_fw_version = key[len(base_fw) :]  # noqa : E203
            if len(original_version.split(".")) == 2:
                base_fw_version = ".".join(base_fw_version.split(".")[:-1])
            versions.append(base_fw_version)
    return sorted(versions, reverse=True)
