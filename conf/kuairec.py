# Standard library imports
from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class DataFrameConfig:
    data_path: str
    used_features: dict


@dataclass(frozen=True)
class VideoDataFrameConfig:
    daily: DataFrameConfig
    category: DataFrameConfig


@dataclass(frozen=True)
class TableConfig:
    interaction: DataFrameConfig
    user: DataFrameConfig
    video: VideoDataFrameConfig


@dataclass(frozen=True)
class LogDataPropensityConfig:
    data_path: str
    train_val_test_ratio: Tuple[float]
    density: float
    behavior_policy: str
    exposure_bias: float


@dataclass(frozen=True)
class ModelConfig:
    n_factors: int
    reg: float
    batch_size: int
    lr: dict


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    seed: int
    data_logging_settings: LogDataPropensityConfig
    tables: TableConfig
    pow_used: float
    is_search_params: bool
    model_params: ModelConfig


@dataclass(frozen=True)
class Config:
    setting: ExperimentConfig
