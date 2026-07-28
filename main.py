from floor_segmentation import logger
from floor_segmentation.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
#from floor_segmentation.pipeline.stage_02_prepare_base_model import PrepareBaseModelTrainingPipeline
#from floor_segmentation.pipeline.stage_03_model_trainer import ModelTrainingPipeline
#from floor_segmentation.pipeline.stage_04_evaluation import EvaluationPipeline



STAGE_NAME = "Data Ingestion stage"


try:
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    obj = DataIngestionTrainingPipeline()
    obj.main()
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise e
