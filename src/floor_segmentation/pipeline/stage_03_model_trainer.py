from floor_segmentation.config.configuration import ConfigurationManager
from floor_segmentation.components.model_trainer import ModelTrainer


STAGE_NAME = "Model Trainer stage"


class ModelTrainerTrainingPipeline:

    def __init__(self):
        pass

    def main(self, checkpoint_path=None):

        # Load the project configuration.
        config = ConfigurationManager()

        # Load the model trainer configuration.
        model_trainer_config = (
            config.get_model_trainer_config()
        )

        # Create the model trainer.
        model_trainer = ModelTrainer(
            config=model_trainer_config
        )

        # Start training with or without checkpoint.
        model_trainer.train(
            checkpoint_path=checkpoint_path
        )