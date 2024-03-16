from dataclasses import dataclass


@dataclass(frozen=True)
class LogDataConfig:
    val_ratio: float


@dataclass(frozen=True)
class DataFrameConfig:
    data_path: str
    txt_path: str


@dataclass(frozen=True)
class TableConfig:
    train: DataFrameConfig
    test: DataFrameConfig
    propensities: DataFrameConfig
    user_features: DataFrameConfig
    item_features: DataFrameConfig


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
    data_logging_settings: LogDataConfig
    tables: TableConfig
    pow_used: float
    is_search_params: bool
    model_params: ModelConfig


@dataclass(frozen=True)
class Config:
    setting: ExperimentConfig
