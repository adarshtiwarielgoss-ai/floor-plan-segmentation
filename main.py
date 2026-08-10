from floor_segmentation.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from floor_segmentation.pipeline.stage_02_data_validation import DataValidationTrainingPipeline
from floor_segmentation.pipeline.stage_03_model_trainer import ModelTrainerTrainingPipeline
from floor_segmentation.utils.hf_syncer import HuggingFaceSyncer
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

    # 1. path where models will save
    SAVE_DIR = "artifacts/model_trainer/room_segmentation" 

    # 2. Background Thread Start
    hf_syncer = HuggingFaceSyncer(save_dir=SAVE_DIR, interval=180)
    hf_syncer.start()

    # 3. Training Execution
    obj = ModelTrainerTrainingPipeline()
    obj.main()

    # 4. Training complete then  final upload & stop
    hf_syncer.stop()

    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")

except Exception as e:
    logger.exception(e)
    raise e