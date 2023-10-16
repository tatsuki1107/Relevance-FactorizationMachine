from dataclasses import dataclass
from typing import Tuple, Union


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
class ParamRangeConfig:
    min: Union[int, float]
    max: Union[int, float]


@dataclass(frozen=True)
class FactorizationMachineConfig:
    n_epochs: ParamRangeConfig
    n_factors: ParamRangeConfig
    lr: ParamRangeConfig
    batch_size: ParamRangeConfig
    clipping: ParamRangeConfig


@dataclass(frozen=True)
class LogisticMatrixFactorizationConfig:
    n_epochs: ParamRangeConfig
    n_factors: ParamRangeConfig
    lr: ParamRangeConfig
    reg: ParamRangeConfig
    batch_size: ParamRangeConfig
    clipping: ParamRangeConfig


@dataclass(frozen=True)
class ModelConfig:
    FM: FactorizationMachineConfig
    MF: LogisticMatrixFactorizationConfig


@dataclass(frozen=True)
class ExperimentConfig:
    seed: int
    data_logging_settings: LogDataPropensityConfig
    tables: TableConfig
    is_search_params: bool
    model_param_range: ModelConfig
