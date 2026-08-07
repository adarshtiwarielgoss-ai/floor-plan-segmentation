import json
import os

from ultralytics import YOLO

from floor_segmentation import logger
from floor_segmentation.entity.config_entity import ModelEvaluationConfig


class ModelEvaluation:

    def __init__(self, config: ModelEvaluationConfig):
        self.config = config

    def evaluate(self):

        logger.info("Loading trained model...")

        model = YOLO(str(self.config.model_path))

        logger.info("Running model evaluation...")

        metrics = model.val(
            data=str(self.config.data_yaml)
        )

        return metrics

    def save_metrics(self, metrics):

        logger.info("Saving evaluation metrics...")

        metric_dict = {

            "mIoU": float(metrics.miou),
            "PixelAccuracy": float(metrics.pixel_accuracy),
            "Fitness": float(metrics.fitness),

            "PerClassIoU": metrics.per_class_iou.tolist(),
            "PerClassPixelAccuracy": metrics.per_class_pixel_accuracy.tolist(),

            "Speed": {
                "Preprocess(ms)": float(metrics.speed["preprocess"]),
                "Inference(ms)": float(metrics.speed["inference"]),
                "Loss(ms)": float(metrics.speed["loss"]),
                "Postprocess(ms)": float(metrics.speed["postprocess"])
            },

            "Results": metrics.results_dict
        }

        with open(self.config.metric_file_name, "w") as f:
            json.dump(metric_dict, f, indent=4)

        logger.info(f"Metrics saved at: {self.config.metric_file_name}")

        csv_path = os.path.join(self.config.root_dir, "metrics.csv")
        metrics.to_csv(csv_path)

        logger.info(f"CSV saved at: {csv_path}")

        json_path = os.path.join(
            self.config.root_dir,
            "metrics_ultralytics.json"
        )

        with open(json_path, "w") as f:
            f.write(metrics.to_json())

        logger.info(f"Ultralytics JSON saved at: {json_path}")