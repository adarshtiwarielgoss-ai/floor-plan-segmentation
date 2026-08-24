from floor_segmentation.pipeline.stage_01_data_ingestion import (
    DataIngestionTrainingPipeline
)

from floor_segmentation.pipeline.stage_02_data_validation import (
    DataValidationTrainingPipeline
)

from floor_segmentation.pipeline.stage_03_model_trainer import (
    ModelTrainerTrainingPipeline
)

from floor_segmentation import logger

from floor_segmentation.utils.auto_zip_and_download import (
    auto_zip_and_download
)


# ==============================================================
# Stage 1: Data Ingestion
# ==============================================================

STAGE_NAME = "Data Ingestion stage"

try:

    logger.info(
        f">>>>>> stage {STAGE_NAME} started <<<<<<"
    )

    obj = DataIngestionTrainingPipeline()

    obj.main()

    logger.info(
        f">>>>>> stage {STAGE_NAME} completed "
        f"<<<<<<\n\nx==========x"
    )

except Exception as e:

    logger.exception(e)

    raise e


# ==============================================================
# Stage 2: Data Validation
# ==============================================================

STAGE_NAME = "Data Validation stage"

try:

    logger.info(
        f">>>>>> stage {STAGE_NAME} started <<<<<<"
    )

    obj = DataValidationTrainingPipeline()

    obj.main()

    logger.info(
        f">>>>>> stage {STAGE_NAME} completed "
        f"<<<<<<\n\nx==========x"
    )

except Exception as e:

    logger.exception(e)

    raise e


# ==============================================================
# Stage 3: Model Trainer
# ==============================================================

STAGE_NAME = "Model Trainer stage"

try:

    logger.info(
        f">>>>>> stage {STAGE_NAME} started <<<<<<"
    )

    # The model trainer pipeline handles Hugging Face connection,
    # checkpoint restoration, background synchronization,
    # training, and final synchronization.
    obj = ModelTrainerTrainingPipeline()

    obj.main()

    # Create an archive of the final training artifacts.
    auto_zip_and_download(
        folder_path="artifacts/model_trainer",
        zip_name="model_trainer_results"
    )

    logger.info(
        f">>>>>> stage {STAGE_NAME} completed "
        f"<<<<<<\n\nx==========x"
    )

except Exception as e:

    logger.exception(e)

    raise e