from sagemaker.sklearn.processing import SKLearnProcessor


class SKLearnProcessorBuilder:
    """Handles Amazon SageMaker processing tasks for jobs using scikit-learn."""

    def __init__(self) -> None:
        self._instance = None

    def __call__(
        self,
        framework_version,
        role,
        instance_type,
        instance_count,
        sagemaker_session,
        volume_size_in_gb=30,
        volume_kms_key=None,
        output_kms_key=None,
        max_runtime_in_seconds=None,
        base_job_name=None,
        env=None,
        tags=None,
        network_config=None,
        **_ignored,
    ) -> SKLearnProcessor:
        if not self._instance:
            skLearn_processor = SKLearnProcessor(
                framework_version=framework_version,
                role=role,
                instance_type=instance_type,
                instance_count=instance_count,
                sagemaker_session=sagemaker_session,
                volume_size_in_gb=volume_size_in_gb,
                volume_kms_key=volume_kms_key,
                output_kms_key=output_kms_key,
                max_runtime_in_seconds=max_runtime_in_seconds,
                base_job_name=base_job_name,
                env=env,
                tags=tags,
                network_config=network_config,
            )
            self._instance = skLearn_processor
            print(
                f"***************************************************************{skLearn_processor}"
            )
        return self._instance
