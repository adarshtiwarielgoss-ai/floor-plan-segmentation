from ultralytics import YOLO

from floor_segmentation import logger
from floor_segmentation.entity.config_entity import PredictionConfig


class Prediction:

    def __init__(self, config: PredictionConfig):

        self.config = config

        logger.info("Loading model...")

        self.model = YOLO(str(self.config.model_path))

        logger.info("Model loaded successfully.")


    def predict(self, image_path):

        logger.info(f"Running prediction on : {image_path}")

        results = self.model.predict(

            source=image_path,

            imgsz=self.config.imgsz,

            conf=self.config.conf,

            device=self.config.device,

            save=self.config.save,

            verbose=False

        )

        logger.info("Prediction completed.")

        return results