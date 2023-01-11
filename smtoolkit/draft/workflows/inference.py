from sagemaker.workflow.pipeline import Pipeline
from smtools import logger
from smtools.ext.core.builder import InferenceBuilder
from smtools.ext.core.singleton import ml_platform


class Inference:
    def __init__(self) -> None:
        self._builder = None
        self._config = None

    @property
    def config(self):
        return self._config

    @config.setter
    def config(self, config) -> None:
        self._config = config

    @property
    def builder(self) -> InferenceBuilder:
        return self._builder

    @builder.setter
    def builder(self, builder: InferenceBuilder) -> None:
        """
        The Director works with any builder instance that the client code passes
        to it. This way, the client code may alter the final type of the newly
        assembled product.
        """
        self._builder = builder

    """
    The Director can construct several product variations using the same
    building steps.
    """

    def build_pipeline(self) -> None:
        logger.info("build_pipeline")

        logger.info("produce paramters")
        self.builder.produce_paramters(self._config.parameters)

        logger.info("produce pre_process_data")
        self.builder.produce_pre_process_data(self._config)

        logger.info("produce produce_inference")
        self.builder.produce_inference(self._config)

        logger.info("produce post_process_data")
        self.builder.produce_post_process_data(self._config)

        # logger.info("produce produce_evaluation")
        # self.builder.produce_evaluation(self._config)

        # logger.info("produce produce_condition")
        # self.builder.produce_condition(self._config)

    def get_pipeline(self):
        self.build_pipeline()

        _pipeline = self.builder.pipeline
        pipeline = Pipeline(
            name=self._config.parameters.pipeline_name,
            parameters=_pipeline.parameters,
            steps=_pipeline.steps,
        )

        return pipeline

    def publish_pipeline(self, start=False):
        self.build_pipeline()

        _pipeline = self.builder.pipeline
        pipeline = Pipeline(
            name=self._config.parameters.pipeline_name,
            parameters=_pipeline.parameters,
            steps=_pipeline.steps,
        )

        logger.info("pipeline.upsert")
        logger.info(f"role_arn=ml_platform.role {ml_platform.role}")
        logger.info(f"tags=ml_platform.tags {ml_platform.tags}")
        pipeline.upsert(role_arn=ml_platform.role, tags=ml_platform.tags)

        if start:
            pipeline.start()

        return pipeline
