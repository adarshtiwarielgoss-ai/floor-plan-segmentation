from floor_segmentation.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from floor_segmentation.pipeline.stage_02_data_validation import DataValidationTrainingPipeline
from floor_segmentation.pipeline.stage_03_model_trainer import ModelTrainerTrainingPipeline

from floor_segmentation.utils.hf_syncer import HuggingFaceSyncer
from floor_segmentation import logger
from floor_segmentation.utils.auto_zip_and_download import auto_zip_and_download


STAGE_NAME = "Data Ingestion stage"

try:

    logger.info(
        f">>>>>> stage {STAGE_NAME} started <<<<<<"
    )

    obj = DataIngestionTrainingPipeline()
    obj.main()

    logger.info(
        f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x"
    )

except Exception as e:

    logger.exception(e)
    raise e


# ==============================================================
# DATA VALIDATION
# ==============================================================

STAGE_NAME = "Data Validation stage"

try:

    logger.info(
        f">>>>>> stage {STAGE_NAME} started <<<<<<"
    )

    obj = DataValidationTrainingPipeline()
    obj.main()

    logger.info(
        f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x"
    )

except Exception as e:

    logger.exception(e)
    raise e


# ==============================================================
# MODEL TRAINER
# ==============================================================

STAGE_NAME = "Model Trainer stage"

try:

    logger.info(
        f">>>>>> stage {STAGE_NAME} started <<<<<<"
    )

    SAVE_DIR = "artifacts/model_trainer"

    # ==========================================================
    # 1. CREATE HF SYNCER
    # ==========================================================

    hf_syncer = HuggingFaceSyncer(
        save_dir=SAVE_DIR,
        interval=180
    )

    # ==========================================================
    # 2. CHECK AND RESTORE LAST CHECKPOINT
    # ==========================================================

    logger.info(
        "Checking HuggingFace for an existing training checkpoint..."
    )

    resume_checkpoint = hf_syncer.restore_last_checkpoint()

    if resume_checkpoint:

        logger.info(
            "=================================================="
        )

        logger.info(
            "Existing checkpoint found."
        )

        logger.info(
            f"Resume checkpoint: {resume_checkpoint}"
        )

        logger.info(
            "Training will RESUME from the checkpoint."
        )

        logger.info(
            "=================================================="
        )

    else:

        logger.info(
            "=================================================="
        )

        logger.info(
            "No existing checkpoint found on HuggingFace."
        )

        logger.info(
            "Training will start from SCRATCH."
        )

        logger.info(
            "=================================================="
        )

    # ==========================================================
    # 3. START BACKGROUND HF SYNCHRONIZATION
    # ==========================================================

    hf_syncer.start()

    # ==========================================================
    # 4. EXECUTE TRAINING PIPELINE
    # ==========================================================

    obj = ModelTrainerTrainingPipeline(
        resume_checkpoint=resume_checkpoint
    )

    obj.main()

    # ==========================================================
    # 5. STOP BACKGROUND HF SYNCHRONIZATION
    # ==========================================================

    hf_syncer.stop()

    # ==========================================================
    # 6. ZIP MODEL ARTIFACTS AND TRIGGER DOWNLOAD
    # ==========================================================

    auto_zip_and_download(
        folder_path=SAVE_DIR,
        zip_name="model_trainer_results"
    )

    logger.info(
        f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x"
    )

except Exception as e:

    logger.exception(e)
    raise e