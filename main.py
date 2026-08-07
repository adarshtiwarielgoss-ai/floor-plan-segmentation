from floor_segmentation.utils.common import configure_ultralytics
configure_ultralytics()
from floor_segmentation.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from floor_segmentation.pipeline.stage_02_data_validation import DataValidationTrainingPipeline
from floor_segmentation.pipeline.stage_03_model_trainer import ModelTrainerTrainingPipeline
from floor_segmentation.pipeline.stage_04_model_evaluation import ModelEvaluationPipeline
from floor_segmentation.pipeline.stage_05_model_pusher import ModelPusherPipeline

from floor_segmentation import logger


STAGE_NAME = "Data Ingestion stage"

try:
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")

    obj = DataIngestionTrainingPipeline()
    obj.main()

    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")

except Exception as e:
    logger.exception(e)
    raise e



STAGE_NAME = "Data Validation stage"

try:
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")

    obj = DataValidationTrainingPipeline()
    obj.main()

    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")

except Exception as e:
    logger.exception(e)
    raise e



STAGE_NAME = "Model Trainer stage"

try:
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")

    obj = ModelTrainerTrainingPipeline()
    obj.main()

    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")

except Exception as e:
    logger.exception(e)
    raise e




STAGE_NAME = "Model Evaluation stage"

try:

    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")

    obj = ModelEvaluationPipeline()

    obj.main()

    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")

except Exception as e:
    logger.exception(e)
    raise e


STAGE_NAME = "Model Pusher stage"

try:

    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")

    obj = ModelPusherPipeline()

    obj.main()

    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")

except Exception as e:

    logger.exception(e)

    raise e