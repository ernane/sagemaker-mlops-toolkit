import os

from sagemaker.dataset_definition.inputs import (
    AthenaDatasetDefinition,
    DatasetDefinition,
)
from sagemaker.inputs import TrainingInput
from sagemaker.model import Model
from sagemaker.model_metrics import MetricsSource, ModelMetrics
from sagemaker.processing import FrameworkProcessor, ProcessingInput, ProcessingOutput
from sagemaker.transformer import Transformer
from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.workflow.conditions import ConditionGreaterThanOrEqualTo
from sagemaker.workflow.execution_variables import ExecutionVariables
from sagemaker.workflow.functions import Join, JsonGet
from sagemaker.workflow.model_step import ModelStep
from sagemaker.workflow.parameters import (
    ParameterFloat,
    ParameterInteger,
    ParameterString,
)
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.properties import PropertyFile
from sagemaker.workflow.steps import ProcessingStep, TrainingStep, TransformStep
from smtools import base_dir, logger
from smtools.ext.core.abstract import AbstractInference, AbstractTraining
from smtools.ext.core.factory import (
    sagemaker_estimator_factory,
    sagemaker_processor_factory,
)
from smtools.ext.core.singleton import ml_platform
from smtools.ext.settings.config import settings

evaluation_report = PropertyFile(
    name="EvaluationReport",
    output_name="evaluation",
    path="evaluation.json",
)


class TrainingBuilder(AbstractTraining):
    def __init__(self) -> None:
        self.reset()

    def reset(self, pipeline_name="pipeline") -> None:
        self._pipeline = Pipeline(pipeline_name)

    @property
    def pipeline(self) -> Pipeline:
        pipeline = self._pipeline
        self.reset()
        return pipeline

    def produce_paramters(self, parameters) -> None:
        """workflow Parameters"""

        logger.info(f"Add parameter PipelineBucket {parameters.pipeline_bucket}")
        self._pipeline.parameters.append(
            ParameterString(
                name="PipelineBucket", default_value=parameters.pipeline_bucket
            )
        )

        logger.info(f"Add parameter PipelinePrefix {parameters.pipeline_prefix}")
        self._pipeline.parameters.append(
            ParameterString(
                name="PipelinePrefix", default_value=parameters.pipeline_prefix
            )
        )

        logger.info(f"Add parameter InstanceCount {parameters.instance_count}")
        self._pipeline.parameters.append(
            ParameterInteger(
                name="InstanceCount", default_value=parameters.instance_count
            )
        )

        logger.info(f"Add parameter InstanceType {parameters.instance_type}")
        self._pipeline.parameters.append(
            ParameterString(name="InstanceType", default_value=parameters.instance_type)
        )

        logger.info(f"Add parameter TrainInstanceType {parameters.train_instance_type}")
        self._pipeline.parameters.append(
            ParameterString(
                name="TrainInstanceType", default_value=parameters.train_instance_type
            )
        )

        logger.info(f"Add parameter VolumeSizeInGB {parameters.volume_size_in_gb}")
        self._pipeline.parameters.append(
            ParameterInteger(
                name="VolumeSizeInGB", default_value=parameters.volume_size_in_gb
            )
        )

        logger.info(
            f"Add parameter MaxRuntimeInSeconds {parameters.max_runtime_in_seconds}"
        )
        self._pipeline.parameters.append(
            ParameterInteger(
                name="MaxRuntimeInSeconds",
                default_value=parameters.max_runtime_in_seconds,
            )
        )

        logger.info(
            f"Add parameter ModelApprovalStatus {parameters.model_approval_status}"
        )
        self._pipeline.parameters.append(
            ParameterString(
                name="ModelApprovalStatus",
                default_value=parameters.model_approval_status,
            )
        )

        logger.info(f"Add parameter MetricThreshold {parameters.metric_threshold}")
        self._pipeline.parameters.append(
            ParameterFloat(
                name="MetricThreshold", default_value=parameters.metric_threshold
            )
        )

    def produce_pre_process_data(self, config) -> None:
        """Feature Engineering"""

        processor = sagemaker_processor_factory.get_or_create(
            service=config.steps.pre_process_data.processor,
            **config.steps.pre_process_data,
        )
        processor.code_location = f"{processor.code_location}/training/preprocess"
        database = (
            config.steps.pre_process_data.inputs.dataset_definition.athena_dataset_definition.db
        )
        local_path = config.steps.pre_process_data.inputs.dataset_definition.local_path
        output_s3_uri = (
            f"{settings.default_athena_output_s3_uri}/{ml_platform.pipeline_id}"
        )
        destination = (
            f"{settings.default_datasets_output_s3_uri}/{ml_platform.pipeline_id}"
        )

        query_file = os.path.join(
            base_dir,
            config.steps.pre_process_data.source_dir,
            config.steps.pre_process_data.query,
        )

        logger.info("Load query file - {}".format(query_file))
        with open(query_file, "r") as f:
            query_string = f.read()

        processor_args = processor.run(
            source_dir=os.path.join(base_dir, config.steps.pre_process_data.source_dir),
            inputs=[
                ProcessingInput(
                    input_name=config.steps.pre_process_data.inputs.input_name,
                    dataset_definition=DatasetDefinition(
                        local_path=local_path,
                        data_distribution_type="ShardedByS3Key",
                        athena_dataset_definition=AthenaDatasetDefinition(
                            catalog="awsdatacatalog",
                            database=database,
                            query_string=query_string,
                            output_s3_uri=output_s3_uri,
                            output_format="PARQUET",
                        ),
                    ),
                )
            ],
            outputs=[
                ProcessingOutput(
                    output_name=output.output_name,
                    destination=f"{destination}/{output.output_name}",
                    source=output.source,
                )
                for output in config.steps.pre_process_data.outputs
            ],
            code=config.steps.pre_process_data.code,
        )
        step_process = ProcessingStep(
            name=config.steps.pre_process_data.name, step_args=processor_args
        )

        self._pipeline.steps.append(step_process)

    def produce_train(self, config) -> None:
        """Model Training"""

        estimator = sagemaker_estimator_factory.get_or_create(
            service=config.steps.model_training.estimator, **config.steps.model_training
        )

        job_args = estimator.fit(
            inputs={
                "train": TrainingInput(
                    s3_data=self._pipeline.steps[0]
                    .properties.ProcessingOutputConfig.Outputs["train"]
                    .S3Output.S3Uri,
                    content_type="text/csv",
                ),
                "test": TrainingInput(
                    s3_data=self._pipeline.steps[0]
                    .properties.ProcessingOutputConfig.Outputs["test"]
                    .S3Output.S3Uri,
                    content_type="text/csv",
                ),
            },
        )

        step_training = TrainingStep(
            name=config.steps.model_training.name,
            step_args=job_args,
            cache_config=ml_platform.cache_config,
        )

        self._pipeline.steps.append(step_training)

    def produce_evaluation(self, config) -> None:
        """Model Evaluation"""

        evaluation_processor = sagemaker_processor_factory.get_or_create(
            service=config.steps.model_evaluation.processor,
            **config.steps.model_evaluation,
        )
        evaluation_processor.base_job_name = config.steps.model_evaluation.base_job_name
        evaluation_processor.code_location = (
            f"{evaluation_processor.code_location}/traning/evaluation"
        )

        step_args = evaluation_processor.run(
            source_dir=os.path.join(base_dir, config.steps.model_evaluation.source_dir),
            code=config.steps.model_evaluation.code,
            inputs=[
                ProcessingInput(
                    input_name="model_artifacts",
                    source=self._pipeline.steps[
                        1
                    ].properties.ModelArtifacts.S3ModelArtifacts,
                    destination="/opt/ml/processing/model",
                ),
                ProcessingInput(
                    input_name="validation",
                    source=self._pipeline.steps[0]
                    .properties.ProcessingOutputConfig.Outputs["validation"]
                    .S3Output.S3Uri,
                    destination="/opt/ml/processing/validation",
                ),
            ],
            outputs=[
                ProcessingOutput(
                    output_name="evaluation",
                    source="/opt/ml/processing/evaluation",
                    destination=f"{ml_platform.model_artifacts}/evaluation",
                ),
            ],
        )

        step_evaluation = ProcessingStep(
            name=config.steps.model_evaluation.name,
            description="Collection Evaluation",
            step_args=step_args,
            property_files=[evaluation_report],
            cache_config=ml_platform.cache_config,
        )

        self._pipeline.steps.append(step_evaluation)

    def produce_condition(self, config) -> None:
        """
        Model Accuracy
        if metric_threshold
            - Model Create
            - Model Package
        else
            - Execution Failed
        """
        model_metrics = ModelMetrics(
            model_statistics=MetricsSource(
                s3_uri="{}/evaluation.json".format(
                    self._pipeline.steps[2].arguments["ProcessingOutputConfig"][
                        "Outputs"
                    ][0]["S3Output"]["S3Uri"]
                ),
                content_type="application/json",
            )
        )

        estimator = sagemaker_estimator_factory.get_or_create(
            service=config.steps.model_training.estimator, **config.steps.model_training
        )
        step_training = self._pipeline.steps[1]
        model = Model(
            name=config.steps.model_training.model_name,
            image_uri=estimator.training_image_uri(),
            model_data=step_training.properties.ModelArtifacts.S3ModelArtifacts,
            entry_point=estimator.entry_point,
            source_dir=estimator.source_dir,
            role=ml_platform.role,
            code_location=estimator.code_location,
            sagemaker_session=ml_platform.session,
        )

        # model.delete_model()
        # model.create(tags=ml_platform.tags)

        model_registry_args = model.register(
            content_types=config.steps.model_registry.content_types,
            response_types=["application/json"],
            inference_instances=["ml.t2.medium", "ml.m5.xlarge"],
            transform_instances=["ml.m5.xlarge"],
            model_package_group_name=config.steps.model_registry.model_group_name,
            approval_status=config["parameters"].model_approval_status,
            model_metrics=model_metrics,
            domain="MACHINE_LEARNING",
            task="CLASSIFICATION",
        )

        step_register = ModelStep(name="", step_args=model_registry_args)

        step_evaluation = self._pipeline.steps[2]

        cond_lte = ConditionGreaterThanOrEqualTo(
            left=JsonGet(
                step_name=step_evaluation.name,
                property_file=evaluation_report,
                json_path="binary_classification_metrics.auc.value",
            ),
            right=config.parameters.metric_threshold,
        )

        step_condition = ConditionStep(
            name="CheckEvaluation",
            conditions=[cond_lte],
            if_steps=[step_register],
            else_steps=[],
        )
        self._pipeline.steps.append(step_condition)


class InferenceBuilder(AbstractInference):
    def __init__(self) -> None:
        self.reset()

    def reset(self, pipeline_name="pipeline") -> None:
        self._pipeline = Pipeline(pipeline_name)

    @property
    def pipeline(self) -> Pipeline:
        pipeline = self._pipeline
        self.reset()
        return pipeline

    def produce_paramters(self, parameters) -> None:
        """workflow Parameters"""

        logger.info(f"Add parameter PipelineBucket {parameters.pipeline_bucket}")
        self._pipeline.parameters.append(
            ParameterString(
                name="PipelineBucket", default_value=parameters.pipeline_bucket
            )
        )

        logger.info(f"Add parameter PipelinePrefix {parameters.pipeline_prefix}")
        self._pipeline.parameters.append(
            ParameterString(
                name="PipelinePrefix", default_value=parameters.pipeline_prefix
            )
        )

        logger.info(f"Add parameter InstanceCount {parameters.instance_count}")
        self._pipeline.parameters.append(
            ParameterInteger(
                name="InstanceCount", default_value=parameters.instance_count
            )
        )

        logger.info(f"Add parameter InstanceType {parameters.instance_type}")
        self._pipeline.parameters.append(
            ParameterString(name="InstanceType", default_value=parameters.instance_type)
        )

        logger.info(f"Add parameter VolumeSizeInGB {parameters.volume_size_in_gb}")
        self._pipeline.parameters.append(
            ParameterInteger(
                name="VolumeSizeInGB", default_value=parameters.volume_size_in_gb
            )
        )

        logger.info(
            f"Add parameter MaxRuntimeInSeconds {parameters.max_runtime_in_seconds}"
        )
        self._pipeline.parameters.append(
            ParameterInteger(
                name="MaxRuntimeInSeconds",
                default_value=parameters.max_runtime_in_seconds,
            )
        )

        logger.info(
            f"Add parameter ResultsDatabaseName {parameters.results_database_name}"
        )
        self._pipeline.parameters.append(
            ParameterString(
                name="ResultsDatabaseName",
                default_value=parameters.results_database_name,
            )
        )

        logger.info(
            f"Add parameter ResultsTable1Name {parameters.results_table_1_name}"
        )
        self._pipeline.parameters.append(
            ParameterString(
                name="ResultsTable1Name", default_value=parameters.results_table_1_name
            )
        )

        logger.info(f"Add parameter ResultsPath1URI {parameters.results_path_1_uri}")
        self._pipeline.parameters.append(
            ParameterString(
                name="ResultsPath1URI", default_value=parameters.results_path_1_uri
            )
        )

        logger.info(
            f"Add parameter ResultsTable2Name {parameters.results_table_2_name}"
        )
        self._pipeline.parameters.append(
            ParameterString(
                name="ResultsTable2Name", default_value=parameters.results_table_2_name
            )
        )

        logger.info(f"Add parameter ResultsPath2URI {parameters.results_path_2_uri}")
        self._pipeline.parameters.append(
            ParameterString(
                name="ResultsPath2URI", default_value=parameters.results_path_2_uri
            )
        )

    def produce_pre_process_data(self, config) -> None:
        """Feature Engineering"""

        processor = sagemaker_processor_factory.get_or_create(
            service=config.steps.pre_process_data.processor,
            **config.steps.pre_process_data,
        )
        processor.code_location = f"{processor.code_location}/inference/preprocess"

        database = (
            config.steps.pre_process_data.inputs.dataset_definition.athena_dataset_definition.db
        )
        local_path = config.steps.pre_process_data.inputs.dataset_definition.local_path
        output_s3_uri = (
            f"{settings.default_athena_output_s3_uri}/{ml_platform.pipeline_id}"
        )
        # destination = f"{settings.default_datasets_output_s3_uri}/{ml_platform.pipeline_id}"

        query_file = os.path.join(
            base_dir,
            config.steps.pre_process_data.source_dir,
            config.steps.pre_process_data.query,
        )

        logger.info("Load query file - {}".format(query_file))
        with open(query_file, "r") as f:
            query_string = f.read()

        processor_args = processor.run(
            source_dir=os.path.join(base_dir, config.steps.pre_process_data.source_dir),
            inputs=[
                ProcessingInput(
                    input_name=config.steps.pre_process_data.inputs.input_name,
                    dataset_definition=DatasetDefinition(
                        local_path=local_path,
                        data_distribution_type="ShardedByS3Key",
                        athena_dataset_definition=AthenaDatasetDefinition(
                            catalog="awsdatacatalog",
                            database=database,
                            query_string=query_string,
                            output_s3_uri=output_s3_uri,
                            output_format="PARQUET",
                        ),
                    ),
                )
            ],
            outputs=[
                ProcessingOutput(
                    output_name=output.output_name,
                    destination=Join(
                        on="",
                        values=[
                            settings.default_datasets_output_s3_uri,
                            "/inference/",
                            ExecutionVariables.START_DATETIME,
                            "/output/preprocessing",
                        ],
                    ),
                    source=output.source,
                )
                for output in config.steps.pre_process_data.outputs
            ],
            code=config.steps.pre_process_data.code,
        )
        step_process = ProcessingStep(
            name=config.steps.pre_process_data.name, step_args=processor_args
        )

        self._pipeline.steps.append(step_process)

    def produce_inference(self, config) -> None:
        """Model Inference"""

        model_packages = ml_platform.sm_client.list_model_packages(
            ModelPackageGroupName=config.parameters.name
        )

        latest_model_version_arn = model_packages["ModelPackageSummaryList"][0][
            "ModelPackageArn"
        ]

        model_package = ml_platform.sm_client.describe_model_package(
            ModelPackageName=latest_model_version_arn
        )

        image_uri = model_package["InferenceSpecification"]["Containers"][0]["Image"]
        model_data = model_package["InferenceSpecification"]["Containers"][0][
            "ModelDataUrl"
        ]
        sourcedir = model_package["InferenceSpecification"]["Containers"][0][
            "Environment"
        ]["SAGEMAKER_SUBMIT_DIRECTORY"]
        environment = model_package["InferenceSpecification"]["Containers"][0][
            "Environment"
        ]

        model = Model(
            name=config.parameters.name,
            image_uri=image_uri,
            model_data=model_data,
            sagemaker_session=ml_platform.sm_session,
            role=ml_platform.role,
            source_dir=sourcedir,
            env=model_package["InferenceSpecification"]["Containers"][0]["Environment"],
        )
        model.delete_model()
        model.create(tags=ml_platform.tags)

        transformer = Transformer(
            model_name=model.name,
            instance_count=config.steps.predict.transformer.instance_count,
            instance_type=config.steps.predict.transformer.instance_type,
            assemble_with=config.steps.predict.transformer.assemble_with,
            max_payload=config.steps.predict.transformer.max_payload,
            accept=config.steps.predict.transformer.accept,
            env=environment,
            output_path=Join(
                on="",
                values=[
                    settings.default_datasets_output_s3_uri,
                    "/inference/",
                    ExecutionVariables.START_DATETIME,
                    "/batch_transform/output",
                ],
            ),
            sagemaker_session=ml_platform.session,
        )

        step_process = self._pipeline.steps[0]
        batch_data = step_process.properties.ProcessingOutputConfig.Outputs[
            config.steps.predict.inputs.input_name
        ].S3Output.S3Uri

        transformer_args = transformer.transform(
            data=batch_data,
            split_type=config.steps.predict.transformer.args.split_type,
            content_type=config.steps.predict.transformer.args.content_type,
            input_filter=config.steps.predict.transformer.args.input_filter,
            join_source=config.steps.predict.transformer.args.join_source,
        )

        step_transform = TransformStep(
            name=config.steps.predict.name, step_args=transformer_args
        )

        self._pipeline.steps.append(step_transform)

    def produce_post_process_data(self, config) -> None:
        """Feature Engineering"""

        processor: FrameworkProcessor = sagemaker_processor_factory.get_or_create(
            service=config.steps.post_process_data.processor,
            **config.steps.post_process_data,
        )
        processor.code_location = f"{processor.code_location}/inference/postprocess"

        processor_args = processor.run(
            source_dir=os.path.join(
                base_dir, config.steps.post_process_data.source_dir
            ),
            inputs=[
                ProcessingInput(
                    input_name="batch_transform_output",
                    source=Join(
                        on="",
                        values=[
                            settings.default_datasets_output_s3_uri,
                            "/inference/",
                            ExecutionVariables.START_DATETIME,
                            "/batch_transform/output",
                        ],
                    ),
                    destination="/opt/ml/processing/input/batch_data",
                )
            ],
            outputs=[
                ProcessingOutput(
                    output_name="batch_data",
                    destination=Join(
                        on="",
                        values=[
                            settings.default_datasets_output_s3_uri,
                            "/inference/",
                            ExecutionVariables.START_DATETIME,
                            "/output/postprocessing",
                        ],
                    ),
                    source="/opt/ml/processing/batch_data",
                )
                # for output in config.steps.post_process_data.outputs
            ],
            code=config.steps.post_process_data.code,
            arguments=[
                "--results-database-name",
                config.parameters.results_database_name,
                "--results-table-1-name",
                config.parameters.results_table_1_name,
                "--results-path-1-uri",
                config.parameters.results_path_1_uri,
                "--results-table-2-name",
                config.parameters.results_table_2_name,
                "--results-path-2-uri",
                config.parameters.results_path_2_uri,
            ],
        )

        step_postprocess = ProcessingStep(
            name="PostProcessData",
            step_args=processor_args,
            depends_on=[self._pipeline.steps[1]],
            retry_policies=[
                self.default_resource_limit_retry_policy,
            ],
        )

        self._pipeline.steps.append(step_postprocess)
