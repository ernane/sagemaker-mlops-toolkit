from typing import Dict, List, Optional, Union

from sagemaker.network import NetworkConfig
from sagemaker.session import Session
from sagemaker.workflow.entities import PipelineVariable
from sagemaker.xgboost import XGBoostProcessor


class XGBoostProcessorBuilder:
    """Handles Amazon SageMaker processing tasks for jobs using XGBoost containers."""

    def __init__(self) -> None:
        self._instance = None

    def __call__(
        self,
        framework_version: str,  # New arg
        role: str,
        instance_count: Union[int, PipelineVariable],
        instance_type: Union[str, PipelineVariable],
        py_version: str = "py3",  # New kwarg
        image_uri: Optional[Union[str, PipelineVariable]] = None,
        command: Optional[List[str]] = None,
        volume_size_in_gb: Union[int, PipelineVariable] = 30,
        volume_kms_key: Optional[Union[str, PipelineVariable]] = None,
        output_kms_key: Optional[Union[str, PipelineVariable]] = None,
        code_location: Optional[str] = None,  # New arg
        max_runtime_in_seconds: Optional[Union[int, PipelineVariable]] = None,
        base_job_name: Optional[str] = None,
        sagemaker_session: Optional[Session] = None,
        env: Optional[Dict[str, Union[str, PipelineVariable]]] = None,
        tags: Optional[List[Dict[str, Union[str, PipelineVariable]]]] = None,
        network_config: Optional[NetworkConfig] = None,
        **_ignored,
    ) -> XGBoostProcessor:
        if not self._instance:
            xgboost_processor = XGBoostProcessor(
                role=role,
                instance_type=instance_type,
                framework_version=framework_version,
                instance_count=instance_count,
                sagemaker_session=sagemaker_session,
                py_version=py_version,
                image_uri=image_uri,
                command=command,
                volume_size_in_gb=volume_size_in_gb,
                volume_kms_key=volume_kms_key,
                output_kms_key=output_kms_key,
                code_location=code_location,
                max_runtime_in_seconds=max_runtime_in_seconds,
                base_job_name=base_job_name,
                env=env,
                tags=tags,
                network_config=network_config,
            )
            self._instance = xgboost_processor
        return self._instance
