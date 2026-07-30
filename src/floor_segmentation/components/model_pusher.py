import json
import shutil
from datetime import datetime
from pathlib import Path

from floor_segmentation import logger
from floor_segmentation.entity.config_entity import ModelPusherConfig


class ModelPusher:

    def __init__(self, config: ModelPusherConfig):
        self.config = config


    def get_next_version(self):

        """
        Returns:
            v1
            v2
            v3
            ...
        """

        saved_models = self.config.saved_models_dir

        versions = []

        if saved_models.exists():

            for item in saved_models.iterdir():

                if item.is_dir() and item.name.startswith("v"):

                    try:
                        versions.append(int(item.name[1:]))
                    except:
                        pass

        next_version = 1 if len(versions) == 0 else max(versions) + 1

        return f"v{next_version}"

    def create_version_directory(self):

        version = self.get_next_version()

        version_dir = self.config.saved_models_dir / version

        version_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Created Version Folder : {version_dir}")

        return version_dir, version

    def copy_models(self, version_dir):

        best_model = self.config.trained_model_dir / "best.pt"

        last_model = self.config.trained_model_dir / "last.pt"

        if best_model.exists():

            shutil.copy2(
                best_model,
                version_dir / "best.pt"
            )

            logger.info("Best model copied.")

        else:
            logger.warning("best.pt not found.")

        if last_model.exists():

            shutil.copy2(
                last_model,
                version_dir / "last.pt"
            )

            logger.info("Last model copied.")

        else:
            logger.warning("last.pt not found.")


    def copy_evaluation_files(self, version_dir):

        evaluation_dir = self.config.evaluation_dir

        files = [
            "metrics.json",
            "metrics.csv",
            "metrics_ultralytics.json",
            "summary.json"
        ]

        for file in files:

            source = evaluation_dir / file

            if source.exists():

                shutil.copy2(
                    source,
                    version_dir / file
                )

                logger.info(f"{file} copied.")

            else:

                logger.warning(f"{file} not found.")



def create_model_info(self, version_dir, version):

    info = {

        "model_version": version,

        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),

        "framework": "Ultralytics",

        "model": "YOLO26",

        "task": "Semantic Segmentation",

        "best_model": "best.pt",

        "last_model": "last.pt",

        "metrics_file": "metrics.json"

    }

    with open(version_dir / "model_info.json", "w") as f:

        json.dump(info, f, indent=4)

    logger.info("model_info.json created.")





    def update_latest(self, version_dir):

        latest_dir = self.config.saved_models_dir / "latest"

        if latest_dir.exists():

            shutil.rmtree(latest_dir)

        shutil.copytree(
            version_dir,
            latest_dir
        )

        logger.info("Latest model updated.")




def push_model(self):

    logger.info("Starting Model Pusher...")

    version_dir, version = self.create_version_directory()

    self.copy_models(version_dir)

    self.copy_evaluation_files(version_dir)

    self.create_model_info(version_dir, version)

    self.update_latest(version_dir)

    logger.info("Model Pusher completed successfully.")               