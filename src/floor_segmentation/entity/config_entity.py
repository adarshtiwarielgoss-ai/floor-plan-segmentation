from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_URL: str
    local_data_file: Path
    unzip_dir: Path



@dataclass(frozen=True)
class DataValidationConfig:

    root_dir: Path

    unzip_data_dir: Path

    STATUS_FILE: Path


@dataclass(frozen=True)
class ModelTrainerConfig:

    root_dir: Path
    weight_name: str
    data_yaml: Path

    epochs: int
    patience: int
    imgsz: int
    batch_size: int

    optimizer: str
    lr0: float
    lrf: float
    momentum: float
    weight_decay: float

    cos_lr: bool

    warmup_epochs: int
    warmup_bias_lr: float
    warmup_momentum: float

    mosaic: float
    scale: float
    translate: float
    fliplr: float
    flipud: float

    hsv_h: float
    hsv_s: float
    hsv_v: float

    mixup: float
    copy_paste: float

    overlap_mask: bool
    mask_ratio: int

    val: bool
    plots: bool

    device: str
    workers: int
    amp: bool

    seed: int
    deterministic: bool
    verbose: bool

    project: str
    name: str





@dataclass(frozen=True)
class ModelEvaluationConfig:

    root_dir: Path

    model_path: Path

    data_yaml: Path

    metric_file_name: Path   




@dataclass(frozen=True)
class ModelPusherConfig:

    root_dir: Path

    trained_model_dir: Path

    evaluation_dir: Path

    saved_models_dir: Path  


@dataclass(frozen=True)
class PredictionConfig:

    model_path: Path

    imgsz: int

    conf: float

    device: str

    save: bool       