import hydra
from hydra.core.config_store import ConfigStore
from conf.setting.default import ExperimentConfig
from utils.dataloader import dataloader

cs = ConfigStore.instance()
cs.store(name="config", group="setting", node=ExperimentConfig)


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: ExperimentConfig) -> None:
    interaction_df, features = dataloader(params=cfg.setting)
    print(interaction_df.head())
    print(features.shape)


if __name__ == "__main__":
    main()
