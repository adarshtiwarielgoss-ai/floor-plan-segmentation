from floor_segmentation import logger
from floor_segmentation.config.configuration import ConfigurationManager
from floor_segmentation.components.model_evaluation import ModelEvaluation


STAGE_NAME = "Model Evaluation stage"


class ModelEvaluationPipeline:
    def __init__(self):
        pass

    def main(self):

        config = ConfigurationManager()

        evaluation_config = config.get_model_evaluation_config()

        evaluation = ModelEvaluation(config=evaluation_config)

        metrics = evaluation.evaluate()

        evaluation.save_metrics(metrics)


if __name__ == "__main__":
    try:

        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")

        obj = ModelEvaluationPipeline()
        obj.main()

        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")

    except Exception as e:
        logger.exception(e)
        raise e