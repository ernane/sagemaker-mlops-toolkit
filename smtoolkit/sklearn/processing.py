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
    ) -> SKLearnProcessor:
        if not self._instance:
            skLearn_processor = SKLearnProcessor(
                framework_version=framework_version,
                role=role,
                instance_type=instance_type,
                instance_count=instance_count,
                sagemaker_session=sagemaker_session,
            )
            self._instance = skLearn_processor
        return self._instance
