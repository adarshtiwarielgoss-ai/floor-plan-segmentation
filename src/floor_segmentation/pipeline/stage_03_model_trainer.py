from floor_segmentation.config.configuration import ConfigurationManager
from floor_segmentation.components.model_trainer import ModelTrainer
from floor_segmentation.utils.hf_syncer import HuggingFaceSyncer
from floor_segmentation import logger


STAGE_NAME = "Model Trainer stage"


class ModelTrainerTrainingPipeline:

    def __init__(self):
        pass

    def main(self):

        # Load the project configuration.
        config = ConfigurationManager()

        # Load the model trainer configuration.
        model_trainer_config = (
            config.get_model_trainer_config()
        )

        # Create the Hugging Face synchronization manager.
        hf_syncer = HuggingFaceSyncer(
            save_dir="artifacts/model_trainer",
            interval=180
        )

        # Check Hugging Face for an existing training checkpoint.
        checkpoint_path = (
            hf_syncer.restore_last_checkpoint()
        )

        # Start background Hugging Face synchronization.
        hf_syncer.start()

        try:

            # Create the model trainer.
            model_trainer = ModelTrainer(
                config=model_trainer_config
            )

            # Start training using the existing checkpoint if available.
            model_trainer.train(
                checkpoint_path=checkpoint_path
            )

        finally:

            # Stop the background synchronization and perform final sync.
            hf_syncer.stop()

            # Close the Hugging Face log file handler.
            hf_syncer.close()


if __name__ == "__main__":

    try:

        logger.info(
            f">>>>>> stage {STAGE_NAME} started <<<<<<"
        )

        obj = ModelTrainerTrainingPipeline()

        obj.main()

        logger.info(
            f">>>>>> stage {STAGE_NAME} completed "
            f"<<<<<<\n\nx==========x"
        )

    except Exception as e:

        logger.exception(e)

        raise e