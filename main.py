from floor_segmentation.pipeline.stage_01_data_ingestion import (
    DataIngestionTrainingPipeline
)

from floor_segmentation.pipeline.stage_02_data_validation import (
    DataValidationTrainingPipeline
)

from floor_segmentation.pipeline.stage_03_model_trainer import (
    ModelTrainerTrainingPipeline
)

from floor_segmentation.utils.hf_syncer import (
    HuggingFaceSyncer
)

from floor_segmentation import logger

from floor_segmentation.utils.auto_zip_and_download import (
    auto_zip_and_download
)


# ============================================================
# DATA INGESTION
# ============================================================

STAGE_NAME = "Data Ingestion stage"

try:

    logger.info(
        f">>>>>> stage {STAGE_NAME} started <<<<<<"
    )

    obj = DataIngestionTrainingPipeline()
    obj.main()

    logger.info(
        f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\n"
        f"x==========x"
    )

except Exception as e:

    logger.exception(e)
    raise e


# ============================================================
# DATA VALIDATION
# ============================================================

STAGE_NAME = "Data Validation stage"

try:

    logger.info(
        f">>>>>> stage {STAGE_NAME} started <<<<<<"
    )

    obj = DataValidationTrainingPipeline()
    obj.main()

    logger.info(
        f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\n"
        f"x==========x"
    )

except Exception as e:

    logger.exception(e)
    raise e


# ============================================================
# MODEL TRAINER
# ============================================================

STAGE_NAME = "Model Trainer stage"

hf_syncer = None

try:

    logger.info(
        f">>>>>> stage {STAGE_NAME} started <<<<<<"
    )

    SAVE_DIR = "artifacts/model_trainer"

    # --------------------------------------------------------
    # 1. Create ONE Hugging Face synchronizer
    # --------------------------------------------------------

    hf_syncer = HuggingFaceSyncer(
        save_dir=SAVE_DIR,
        interval=3600,       # 1 hour
    )

    # --------------------------------------------------------
    # 2. Restore checkpoint BEFORE training
    # --------------------------------------------------------

    checkpoint = hf_syncer.restore_checkpoint()

    if checkpoint:

        logger.info(
            "=================================================="
        )

        logger.info(
            "Existing checkpoint found on Hugging Face."
        )

        logger.info(
            f"Checkpoint: {checkpoint}"
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
            "No existing checkpoint found on Hugging Face."
        )

        logger.info(
            "Training will start from SCRATCH."
        )

        logger.info(
            "=================================================="
        )

    # --------------------------------------------------------
    # 3. Start ONE background synchronization
    # --------------------------------------------------------

    hf_syncer.start()

    # --------------------------------------------------------
    # 4. Start model training
    # --------------------------------------------------------

    obj = ModelTrainerTrainingPipeline()

    obj.main(
        checkpoint_path=checkpoint
    )

    # --------------------------------------------------------
    # 5. Stop HF synchronization
    #    Final sync will happen here
    # --------------------------------------------------------

    hf_syncer.stop()

    hf_syncer = None

    # --------------------------------------------------------
    # 6. Create ZIP archive
    # --------------------------------------------------------

    auto_zip_and_download(
        folder_path=SAVE_DIR,
        zip_name="model_trainer_results"
    )

    logger.info(
        f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\n"
        f"x==========x"
    )

except Exception as e:

    logger.exception(e)

    # --------------------------------------------------------
    # Make sure HF sync is stopped if training fails
    # --------------------------------------------------------

    if hf_syncer is not None:

        try:

            hf_syncer.stop()

        except Exception as sync_error:

            logger.error(
                f"Failed to stop Hugging Face synchronizer: "
                f"{sync_error}"
            )

    raise e