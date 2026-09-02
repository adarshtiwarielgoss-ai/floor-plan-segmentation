from floor_segmentation.constants import *
import os
from pathlib import Path
from floor_segmentation.utils.common import (
                                              read_yaml,
                                              create_directories
                                            )

from floor_segmentation.entity.config_entity import (DataIngestionConfig,
                                                DataValidationConfig,
                                                ModelTrainerConfig,
                                                ModelEvaluationConfig,
                                                ModelPusherConfig,
                                                PredictionConfig
                                                )



class ConfigurationManager:
    def __init__(
        self,
        config_filepath=CONFIG_FILE_PATH,
        params_filepath=PARAMS_FILE_PATH,
    ):

        

        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)

        create_directories([self.config.artifacts_root])


    
    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion

        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            source_URL=config.source_URL,
            local_data_file=config.local_data_file,
            unzip_dir=config.unzip_dir 
        )

        return data_ingestion_config



    def get_data_validation_config(self) -> DataValidationConfig:

        config = self.config.data_validation

        create_directories([config.root_dir])

        data_validation_config = DataValidationConfig(
        root_dir=Path(config.root_dir),
        unzip_data_dir=Path(config.unzip_data_dir),
        STATUS_FILE=Path(config.STATUS_FILE),
        )

        return data_validation_config

    


    def get_model_trainer_config(self) -> ModelTrainerConfig:

        config = self.config.model_trainer
        params = self.params

        create_directories([config.root_dir])

        model_trainer_config = ModelTrainerConfig(

            root_dir=Path(config.root_dir),
            weight_name=config.weight_name,

         data_yaml=Path(
            os.path.join(
                self.config.data_ingestion.unzip_dir,
                "room_dataset",
                "data.yaml"
            )
        ),

            epochs=params.EPOCHS,
            patience=params.PATIENCE,

            imgsz=params.IMGSZ,
            batch_size=params.BATCH_SIZE,

            optimizer=params.OPTIMIZER,
            lr0=params.LR0,
            lrf=params.LRF,
            momentum=params.MOMENTUM,
            weight_decay=params.WEIGHT_DECAY,

            cos_lr=params.COS_LR,

            warmup_epochs=params.WARMUP_EPOCHS,
            warmup_bias_lr=params.WARMUP_BIAS_LR,
            warmup_momentum=params.WARMUP_MOMENTUM,

            mosaic=params.MOSAIC,
            scale=params.SCALE,
            translate=params.TRANSLATE,
            fliplr=params.FLIPLR,
            flipud=params.FLIPUD,

            hsv_h=params.HSV_H,
            hsv_s=params.HSV_S,
            hsv_v=params.HSV_V,

            mixup=params.MIXUP,
            copy_paste=params.COPY_PASTE,

            overlap_mask=params.OVERLAP_MASK,
            mask_ratio=params.MASK_RATIO,

            val=params.VAL,
            plots=params.PLOTS,

            device=params.DEVICE,
            workers=params.WORKERS,
            amp=params.AMP,

            seed=params.SEED,
            deterministic=params.DETERMINISTIC,
            verbose=params.VERBOSE,

            name=params.NAME
        )

        return model_trainer_config


    def get_model_evaluation_config(self) -> ModelEvaluationConfig:

        config = self.config.model_evaluation

        create_directories([config.root_dir])

        model_evaluation_config = ModelEvaluationConfig(

        root_dir=Path(config.root_dir),

        model_path=Path(config.model_path),

        data_yaml=Path(config.data_yaml),

        metric_file_name=Path(
            config.root_dir
        ) / config.metric_file_name
        )

        return model_evaluation_config




    def get_model_pusher_config(self) -> ModelPusherConfig:

        config = self.config.model_pusher

        create_directories([
        config.root_dir,
        config.saved_models_dir
        ])

        return ModelPusherConfig(

        root_dir=Path(config.root_dir),

        trained_model_dir=Path(config.trained_model_dir),

        evaluation_dir=Path(config.evaluation_dir),

        saved_models_dir=Path(config.saved_models_dir)

    )



    def get_prediction_config(self) -> PredictionConfig:

        config = self.config.prediction

        return PredictionConfig(

        model_path=Path(config.model_path),

        imgsz=config.imgsz,

        conf=config.conf,

        device=config.device,

        save=config.save

        )