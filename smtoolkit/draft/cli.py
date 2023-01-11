from __future__ import annotations

import argparse
import json

from smtools import logger
from smtools.ext.core.builder import InferenceBuilder, TrainingBuilder
from smtools.ext.settings.config import settings
from smtools.ext.workflows.inference import Inference
from smtools.ext.workflows.training import Training


class SMTools:
    def __init__(self) -> None:
        self.builders = {"training": TrainingBuilder(), "inference": InferenceBuilder()}
        self.workflows = {"training": Training(), "inference": Inference()}

    def create_app(self, service, context):
        app = self.workflows.get(context)
        app.builder = self.builders.get(context)
        app.config = settings.services[service][context]
        return app


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--service", type=str, help="Nome do Servico [modelo]")
    parser.add_argument(
        "-c", "--context", type=str, help="Nome da etapa a ser processada"
    )
    parser.add_argument(
        "-p", "--publish", help="Atualiza o pipeline", action="store_true"
    )
    parser.add_argument("-i", "--init", help="Inicia o pipeline", action="store_true")
    args = parser.parse_args()

    logger.info(args)

    smtools = SMTools()
    app = smtools.create_app(service=args.service, context=args.context)

    if args.publish:
        logger.info(
            json.dumps(
                json.loads(app.publish_pipeline(start=args.init).definition()), indent=2
            )
        )
    else:
        logger.info(json.dumps(json.loads(app.get_pipeline().definition()), indent=2))
