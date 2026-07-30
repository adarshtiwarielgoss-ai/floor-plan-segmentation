from floor_segmentation import logger
from floor_segmentation.config.configuration import ConfigurationManager
from floor_segmentation.components.model_pusher import ModelPusher


STAGE_NAME = "Model Pusher stage"


class ModelPusherPipeline:

    def __init__(self):
        pass

    def main(self):

        config = ConfigurationManager()

        model_pusher_config = config.get_model_pusher_config()

        model_pusher = ModelPusher(config=model_pusher_config)

        model_pusher.push_model()


if __name__ == "__main__":

    try:

        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")

        obj = ModelPusherPipeline()

        obj.main()

        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")

    except Exception as e:

        logger.exception(e)

        raise e