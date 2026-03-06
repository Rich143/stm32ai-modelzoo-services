import tensorflow as tf
from omegaconf import DictConfig

def check_tuner_cfg(cfg):
    tuner_cfg = cfg.tuner
    if tuner_cfg is None:
        raise ValueError("\nNo Tuner CFG found. Please check the 'tuner' section of your configuration file.")

    if tuner_cfg.max_trials is None:
        raise ValueError("\nNo max_trials found. Please check the 'tuner.max_trials' section of your configuration file.")

    if tuner_cfg.executions_per_trial is None:
        raise ValueError("\nNo executions_per_trial found. Please check the 'tuner.executions_per_trial' section of your configuration file.")

def get_early_stopping_cb(cfg: DictConfig):
    if cfg.training.EarlyStopping is not None:
        return tf.keras.callbacks.EarlyStopping(
            monitor=cfg.training.EarlyStopping.monitor,
            mode=cfg.training.EarlyStopping.mode,
            patience=cfg.training.EarlyStopping.patience,
            restore_best_weights=cfg.training.EarlyStopping.restore_best_weights
        )

