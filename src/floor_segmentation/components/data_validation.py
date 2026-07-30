import os
from floor_segmentation import logger
from floor_segmentation.entity.config_entity import DataValidationConfig


class DataValidation:
    def __init__(self, config: DataValidationConfig):
        self.config = config

    def validate_all_files_exist(self) -> bool:
        """
        Validates whether all required files/folders exist.
        """

        validation_status = True

        required_files = [
            "train",
            "valid",
            "data.yaml"
        ]

        dataset_path = self.config.unzip_data_dir

        logger.info(f"Checking dataset at: {dataset_path}")

        for file in required_files:
            file_path = os.path.join(dataset_path, file)

            if not os.path.exists(file_path):
                logger.warning(f"Missing: {file_path}")
                validation_status = False
            else:
                logger.info(f"Found: {file_path}")

        with open(self.config.STATUS_FILE, "w") as f:
            f.write(f"Validation status: {validation_status}")

        if validation_status:
            logger.info("Data Validation Completed Successfully.")
        else:
            logger.error("Data Validation Failed.")

        return validation_status