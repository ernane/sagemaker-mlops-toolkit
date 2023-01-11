import datetime
from threading import Lock

import boto3
import sagemaker
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.workflow.steps import CacheConfig
from smtools.ext.settings.config import settings

boto_session = boto3.Session()


class SingletonMeta(type):
    """
    This is a thread-safe implementation of Singleton.
    """

    _instances = {}

    _lock: Lock = Lock()
    """
    We now have a lock object that will be used to synchronize threads during
    first access to the Singleton.
    """

    def __call__(cls, *args, **kwargs):
        """
        Possible changes to the value of the `__init__` argument do not affect
        the returned instance.
        """
        # Now, imagine that the program has just been launched. Since there's no
        # Singleton instance yet, multiple threads can simultaneously pass the
        # previous conditional and reach this point almost at the same time. The
        # first of them will acquire lock and will proceed further, while the
        # rest will wait here.
        with cls._lock:
            # The first thread to acquire the lock, reaches this conditional,
            # goes inside and creates the Singleton instance. Once it leaves the
            # lock block, a thread that might have been waiting for the lock
            # release may then enter this section. But since the Singleton field
            # is already initialized, the thread won't create a new object.
            if cls not in cls._instances:
                instance = super().__call__(*args, **kwargs)
                cls._instances[cls] = instance
        return cls._instances[cls]


class MLPlatform(metaclass=SingletonMeta):
    session: PipelineSession = None
    role: str = None
    region: str = None
    pipeline_id: str = None
    cache_config: CacheConfig = None
    code_location: str = None
    model_artifacts: str = None
    tags: dict = None
    sm_client = None
    sm_session = None

    """
    We'll use this property to prove that our Singleton really works.
    """

    def __init__(self) -> None:
        self.session = PipelineSession(default_bucket=settings.default_bucket)
        self.role = settings.default_role
        self.region = sagemaker.session.Session().boto_session.region_name
        self.pipeline_id = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S-%f")
        self.cache_config = CacheConfig(enable_caching=True, expire_after="T60m")
        self.code_location = (
            f"{settings.default_code_location_s3_uri}/{self.pipeline_id}"
        )
        self.model_artifacts = (
            f"{settings.default_model_artifacts_s3_uri}/{self.pipeline_id}"
        )
        self.tags = settings.tags
        self.sm_client = boto_session.client("sagemaker")
        self.sm_session = sagemaker.Session(default_bucket=settings.default_bucket)


ml_platform = MLPlatform()
