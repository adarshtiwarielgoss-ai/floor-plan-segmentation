from pathlib import Path

from ultralytics import YOLO

from floor_segmentation import logger
from floor_segmentation.entity.config_entity import ModelTrainerConfig


class ModelTrainer:

    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    def train(self):

        logger.info(f"Loading model: {self.config.weight_name}")

        model = YOLO(self.config.weight_name)

        logger.info("Starting YOLO Training...")

        model.train(
            data=str(self.config.data_yaml),

            epochs=self.config.epochs,
            patience=self.config.patience,

            imgsz=self.config.imgsz,
            batch=self.config.batch_size,

            optimizer=self.config.optimizer,
            lr0=self.config.lr0,
            lrf=self.config.lrf,
            momentum=self.config.momentum,
            weight_decay=self.config.weight_decay,

            cos_lr=self.config.cos_lr,

            warmup_epochs=self.config.warmup_epochs,
            warmup_bias_lr=self.config.warmup_bias_lr,
            warmup_momentum=self.config.warmup_momentum,

            mosaic=self.config.mosaic,
            scale=self.config.scale,
            translate=self.config.translate,
            fliplr=self.config.fliplr,
            flipud=self.config.flipud,

            hsv_h=self.config.hsv_h,
            hsv_s=self.config.hsv_s,
            hsv_v=self.config.hsv_v,

            mixup=self.config.mixup,
            copy_paste=self.config.copy_paste,

            overlap_mask=self.config.overlap_mask,
            mask_ratio=self.config.mask_ratio,

            val=self.config.val,
            plots=self.config.plots,

            device=self.config.device,
            workers=self.config.workers,
            amp=self.config.amp,

            seed=self.config.seed,
            deterministic=self.config.deterministic,
            verbose=self.config.verbose,

            project=str(Path(self.config.root_dir).resolve()),
            name=self.config.name,
            exist_ok=True
        )

        logger.info("Model Training Completed Successfully.")