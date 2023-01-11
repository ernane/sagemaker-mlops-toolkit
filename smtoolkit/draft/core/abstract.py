from abc import ABC, abstractmethod

from sagemaker.workflow.retry import (
    SageMakerJobExceptionTypeEnum,
    SageMakerJobStepRetryPolicy,
)


class AbstractStep(ABC):
    default_resource_limit_retry_policy = SageMakerJobStepRetryPolicy(
        exception_types=[SageMakerJobExceptionTypeEnum.RESOURCE_LIMIT],
        failure_reason_types=[SageMakerJobExceptionTypeEnum.RESOURCE_LIMIT],
        backoff_rate=2.0,
        interval_seconds=60,
        expire_after_mins=60 * 5,  # 5 minutos
    )


class AbstractTraining(AbstractStep):
    """
    The Builder interface specifies methods for creating the different parts of
    the SageMaker Pipeline of Model Training objects.
    """

    @property
    @abstractmethod
    def pipeline(self) -> None:
        pass

    @abstractmethod
    def produce_paramters(self) -> None:
        """workflow Parameters"""
        pass

    @abstractmethod
    def produce_pre_process_data(self) -> None:
        """Feature Engineering"""
        pass

    @abstractmethod
    def produce_train(self) -> None:
        """Model Training"""
        pass

    @abstractmethod
    def produce_evaluation(self) -> None:
        """Model Evaluation"""
        pass

    @abstractmethod
    def produce_condition(self) -> None:
        """
        Model Accuracy
        if metric_threshold
            - Model Create
            - Model Package
        else
            - Execution Failed
        """
        pass


class AbstractInference(AbstractStep):
    """
    The Builder interface specifies methods for creating the different parts of
    the SageMaker Pipeline of Model Inference.
    """

    @property
    @abstractmethod
    def pipeline(self) -> None:
        pass

    @abstractmethod
    def produce_paramters(self) -> None:
        """workflow Parameters"""
        pass

    @abstractmethod
    def produce_pre_process_data(self) -> None:
        """Feature Engineering"""
        pass

    @abstractmethod
    def produce_inference(self) -> None:
        """Model Inference"""
        pass

    @abstractmethod
    def produce_post_process_data(self) -> None:
        """Feature Engineering"""
        pass
