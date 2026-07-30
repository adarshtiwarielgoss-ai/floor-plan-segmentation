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

            # Training
            epochs=self.config.epochs,
            patience=self.config.patience,

            # Image
            imgsz=self.config.imgsz,

            # Batch
            batch=self.config.batch_size,

            # Optimizer
            optimizer=self.config.optimizer,
            lr0=self.config.lr0,
            lrf=self.config.lrf,
            momentum=self.config.momentum,
            weight_decay=self.config.weight_decay,

            # LR Scheduler
            cos_lr=self.config.cos_lr,

            # Warmup
            warmup_epochs=self.config.warmup_epochs,
            warmup_bias_lr=self.config.warmup_bias_lr,
            warmup_momentum=self.config.warmup_momentum,

            # Augmentation
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

            # Segmentation
            overlap_mask=self.config.overlap_mask,
            mask_ratio=self.config.mask_ratio,

            # Validation
            val=self.config.val,
            plots=self.config.plots,

            # Hardware
            device=self.config.device,
            workers=self.config.workers,
            amp=self.config.amp,

            # Misc
            seed=self.config.seed,
            deterministic=self.config.deterministic,
            verbose=self.config.verbose,

            # Output
            project=str(self.config.root_dir),
            name=self.config.name
        )

        logger.info("Model Training Completed Successfully.")