from dataclasses import dataclass
from typing import Tuple, List


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
    behavior_policy: str
    exposure_bias: float


@dataclass
class FactorizationMachineConfig:
    n_epochs: List[int]
    n_factors: List[int]
    scale: List[float]
    lr: List[float]
    batch_size: List[int]
    clipping: List[float]


@dataclass
class ProbabilisticMatrixFactorizationConfig:
    n_epochs: List[int]
    n_factors: List[int]
    scale: List[float]
    lr: List[float]
    reg: List[float]
    batch_size: List[int]
    clipping: List[float]


@dataclass
class ModelConfig:
    FM: FactorizationMachineConfig
    PMF: ProbabilisticMatrixFactorizationConfig


@dataclass
class ExperimentConfig:
    seed: int
    logdata_propensity: LogDataPropensityConfig
    tables: TableConfig
    model: ModelConfig
