from smtoolkit.sklearn.processing import SKLearnProcessorBuilder


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


class AWSProcessing(ObjectFactory):
    def get_or_create(self, processor, **kwargs):
        return self.create(processor, **kwargs)


aws_processing = AWSProcessing()
aws_processing.register_builder("SKLearnProcessor", SKLearnProcessorBuilder())
