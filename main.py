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
# Data Ingestion Stage
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
# Data Validation Stage
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
# Model Trainer Stage
# ==============================================================

STAGE_NAME = "Model Trainer stage"

try:

    logger.info(
        f">>>>>> stage {STAGE_NAME} started <<<<<<"
    )

    # The model trainer pipeline handles Hugging Face,
    # checkpoint restoration, synchronization, and training.
    obj = ModelTrainerTrainingPipeline()

    obj.main()

    # Create the final training artifact archive.
    auto_zip_and_download(
        folder_path="artifacts/model_trainer",
        zip_name="model_trainer_results",
    )

    logger.info(
        f">>>>>> stage {STAGE_NAME} completed "
        f"<<<<<<\n\nx==========x"
    )

except Exception as e:

    logger.exception(e)

    raise e