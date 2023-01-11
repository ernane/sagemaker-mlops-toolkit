import os

import sagemaker
from sagemaker.processing import FrameworkProcessor
from sagemaker.sklearn import SKLearn
from sagemaker.sklearn.processing import SKLearnProcessor
from smtools import base_dir
from smtools.ext.core.singleton import ml_platform


class ObjectFactory:
    def __init__(self):
        self._builders = {}

    @property
    def builders(self):
        return self._builders

    def register_builder(self, key, builder):
        self._builders[key] = builder

    def create(self, key, **kwargs):
        builder = self._builders.get(key)
        if not builder:
            raise ValueError(key)
        return builder(**kwargs)


class SageMakerProcessorFactory(ObjectFactory):
    def get_or_create(self, service, **kwargs):
        return self.create(service, **kwargs)


class SageMakerEstimatorFactory(ObjectFactory):
    def get_or_create(self, service, **kwargs):
        return self.create(service, **kwargs)


class SKLearnProcessorBuilder:
    def __init__(self) -> None:
        self._instance = None

    def __call__(
        self,
        framework_version,
        instance_type,
        instance_count,
        base_job_name,
        **_ignored,
    ) -> SKLearnProcessor:
        if not self._instance:
            self._instance = SKLearnProcessor(
                framework_version=framework_version,
                instance_type=instance_type,
                instance_count=instance_count,
                base_job_name=base_job_name,
                role=ml_platform.role,
                sagemaker_session=ml_platform.session,
            )
        return self._instance


class SKLearnFrameworkProcessorBuilder:
    def __init__(self) -> None:
        self._instance = None

    def __call__(
        self,
        framework_version,
        framework,
        instance_count,
        instance_type,
        base_job_name,
        **_ignored,
    ) -> FrameworkProcessor:
        image_uri = sagemaker.image_uris.retrieve(
            framework=framework, region=ml_platform.region, version=framework_version
        )

        self._instance = FrameworkProcessor(
            estimator_cls=SKLearn,
            framework_version=framework_version,
            image_uri=image_uri,
            role=ml_platform.role,
            instance_count=instance_count,
            instance_type=instance_type,
            command=["python3"],
            # volume_size_in_gb=volume_size_in_gb,
            # max_runtime_in_seconds=max_runtime_in_seconds,
            sagemaker_session=ml_platform.session,
            code_location=ml_platform.code_location,
            base_job_name=base_job_name,
        )
        return self._instance


sagemaker_processor_factory = SageMakerProcessorFactory()
sagemaker_processor_factory.register_builder(
    "SKLearnProcessor", SKLearnProcessorBuilder()
)
sagemaker_processor_factory.register_builder(
    "SKLearnFrameworkProcessor", SKLearnFrameworkProcessorBuilder()
)


class SKLearnEstimatorBuilder:
    def __init__(self) -> None:
        self._instance = None

    def __call__(
        self,
        entry_point,
        source_dir,
        instance_count,
        instance_type,
        framework,
        framework_version,
        metric_definitions,
        base_job_name,
        hyperparameters,
        **_ignored,
    ) -> SKLearn:
        if not self._instance:

            image_uri = sagemaker.image_uris.retrieve(
                framework=framework,
                region=ml_platform.region,
                version=framework_version,
            )

            self._instance = SKLearn(
                entry_point=entry_point,
                source_dir=os.path.join(base_dir, source_dir),
                role=ml_platform.role,
                instance_count=instance_count,
                instance_type=instance_type,
                image_uri=image_uri,
                framework_version=framework_version,
                sagemaker_session=ml_platform.session,
                script_mode=True,
                metric_definitions=metric_definitions,
                use_spot_instances=True,
                max_run=60 * 12,
                max_wait=60 * 12,
                code_location=ml_platform.code_location,
                output_path=ml_platform.model_artifacts,
                base_job_name=base_job_name,
                hyperparameters=hyperparameters,
                # env=env,
            )
        return self._instance


sagemaker_estimator_factory = SageMakerEstimatorFactory()
sagemaker_estimator_factory.register_builder("SKLearn", SKLearnEstimatorBuilder())
