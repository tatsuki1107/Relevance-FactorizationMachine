from dataclasses import dataclass
from typing import Tuple


@dataclass
class InteractionFeatureConfig:
    timestamp: str


@dataclass
class InteractionTableConfig:
    data_path: str
    features: InteractionFeatureConfig


@dataclass
class UserFeatureConfig:
    onehot_feat0: str
    onehot_feat1: str
    onehot_feat2: str
    onehot_feat6: str
    onehot_feat11: str
    onehot_feat12: str
    onehot_feat13: str
    onehot_feat14: str
    register_days: str


@dataclass
class UserTableConfig:
    data_path: str
    features: UserFeatureConfig


@dataclass
class VideoDailyFeatureConfig:
    play_progress: str
    video_duration: str
    like_cnt: str
    share_user_num: str


@dataclass
class VideoDailyTableConfig:
    data_path: str
    features: VideoDailyFeatureConfig


@dataclass
class VideoCategoryFeatureConfig:
    feat: str


@dataclass
class VideoCategoryTableConfig:
    data_path: str
    features: VideoCategoryFeatureConfig


@dataclass
class VideoTableConfig:
    daily: VideoDailyTableConfig
    category: VideoCategoryTableConfig


@dataclass
class TableConfig:
    interaction: InteractionTableConfig
    user: UserTableConfig
    video: VideoTableConfig


@dataclass
class LogDataPropensityConfig:
    data_path: str
    train_val_test_ratio: Tuple[float, float, float]
    density: float
    exposure_bias: float
    behavior_policy: str


@dataclass
class FactorizationMachineConfig:
    n_epochs: int
    n_factors: int
    scale: float
    lr: float
    batch_size: int


@dataclass
class ProbabilisticMatrixFactorizationConfig:
    n_epochs: int
    n_factors: int
    scale: float
    lr: float
    reg: float
    batch_size: int


@dataclass
class ExperimentConfig:
    seed: int
    logdata_propensity: LogDataPropensityConfig
    tables: TableConfig
    fm: FactorizationMachineConfig
    mf: ProbabilisticMatrixFactorizationConfig
