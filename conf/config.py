from dataclasses import dataclass
from typing import Tuple, List


@dataclass(frozen=True)
class InteractionFeatureConfig:
    timestamp: str


@dataclass(frozen=True)
class InteractionTableConfig:
    data_path: str
    used_features: InteractionFeatureConfig


@dataclass(frozen=True)
class UserFeatureConfig:
    user_active_degree: str
    onehot_feat0: str
    onehot_feat1: str
    onehot_feat2: str
    onehot_feat6: str
    onehot_feat11: str
    onehot_feat12: str
    onehot_feat13: str
    onehot_feat14: str
    onehot_feat15: str


@dataclass(frozen=True)
class UserTableConfig:
    data_path: str
    used_features: UserFeatureConfig


@dataclass(frozen=True)
class VideoDailyFeatureConfig:
    play_progress: str
    video_duration: str
    like_cnt: str
    cancel_like_cnt: str
    share_cnt: str
    double_click_cnt: str
    download_cnt: str


@dataclass(frozen=True)
class VideoDailyTableConfig:
    data_path: str
    used_features: VideoDailyFeatureConfig


@dataclass(frozen=True)
class VideoCategoryFeatureConfig:
    feat: str


@dataclass(frozen=True)
class VideoCategoryTableConfig:
    data_path: str
    used_features: VideoCategoryFeatureConfig


@dataclass(frozen=True)
class VideoTableConfig:
    daily: VideoDailyTableConfig
    category: VideoCategoryTableConfig


@dataclass(frozen=True)
class TableConfig:
    interaction: InteractionTableConfig
    user: UserTableConfig
    video: VideoTableConfig


@dataclass(frozen=True)
class LogDataPropensityConfig:
    data_path: str
    train_val_test_ratio: Tuple[float, float, float]
    density: float
    behavior_policy: str
    exposure_bias: float


@dataclass(frozen=True)
class FactorizationMachineConfig:
    n_epochs: List[int]
    n_factors: List[int]
    lr: List[float]
    batch_size: List[int]
    clipping: List[float]


@dataclass(frozen=True)
class LogisticMatrixFactorizationConfig:
    n_epochs: List[int]
    n_factors: List[int]
    lr: List[float]
    reg: List[float]
    batch_size: List[int]
    clipping: List[float]


@dataclass(frozen=True)
class ModelConfig:
    FM: FactorizationMachineConfig
    MF: LogisticMatrixFactorizationConfig


@dataclass(frozen=True)
class ExperimentConfig:
    seed: int
    logdata_propensity: LogDataPropensityConfig
    tables: TableConfig
    is_search_params: bool
    model: ModelConfig
