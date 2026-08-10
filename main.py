from floor_segmentation.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from floor_segmentation.pipeline.stage_02_data_validation import DataValidationTrainingPipeline
from floor_segmentation.pipeline.stage_03_model_trainer import ModelTrainerTrainingPipeline
from floor_segmentation.utils.hf_syncer import HuggingFaceSyncer
from floor_segmentation import logger
from floor_segmentation.utils.auto_zip_and_download import auto_zip_and_download

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

    SAVE_DIR = "artifacts/model_trainer"

    # 1. Start background Hugging Face synchronization
    hf_syncer = HuggingFaceSyncer(save_dir=SAVE_DIR, interval=180)
    hf_syncer.start()

    # 2. Execute training pipeline
    obj = ModelTrainerTrainingPipeline()
    obj.main()

    # 3. Terminate background synchronization thread
    hf_syncer.stop()

    # 4. Zip model artifacts and trigger direct download/open
    auto_zip_and_download(folder_path=SAVE_DIR, zip_name="model_trainer_results")

    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")

except Exception as e:
    logger.exception(e)
    raise e