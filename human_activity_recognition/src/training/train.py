# /*---------------------------------------------------------------------------------------------
#  * Copyright (c) 2022-2023 STMicroelectronics.
#  * All rights reserved.
#  *
#  * This software is licensed under terms that can be found in the LICENSE file in
#  * the root directory of this software component.
#  * If no LICENSE file comes with this software, it is provided AS-IS.
#  *--------------------------------------------------------------------------------------------*/

import os
import sys
from pathlib import Path
from timeit import default_timer as timer
from datetime import timedelta
from typing import Tuple, List, Dict, Optional

from hydra.core.hydra_config import HydraConfig
from munch import DefaultMunch
from omegaconf import DictConfig
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
import mlflow
import keras_tuner

from logs_utils import log_to_file, log_last_epoch_history, LRTensorBoard
from gpu_utils import check_training_determinism
from models_utils import model_summary
from cfg_utils import collect_callback_args
from cfg_utils import get_random_seed
from common_training import set_frozen_layers, set_dropout_rate, get_optimizer
from models_mgt import get_model, get_loss
import lr_schedulers
from evaluate import evaluate_keras_model
from visualize_utils import vis_training_curves
from gmp_tuner import create_build_model as gmp_create_build_model
from data_load_helpers import global_activity_name_to_id

from math import ceil

def load_model_to_train(cfg, model_path=None, num_classes=None) -> tf.keras.Model:
    """
    This function loads the model to train, which can be either a:
    - model from the zoo (ign, gmp, custom) .h5 model
    - a model loaded from a .h5 file with  the training is being resumed
    - or, a .h5 model outside of the model zoo.

    These 3 different cases are mutually exclusive.

    Arguments:
        cfg (DictConfig): a dictionary containing the 'training' section 
                          of the configuration file.
        model_path (str): a path to a .h5 file provided using the 
                          'general.model_path' attribute.

    Return:
        tf.keras.Model: a keras model.
    """

    if cfg.model:
        # Model from the zoo
        model = get_model(cfg=cfg.model,
                          num_classes=num_classes,
                          dropout=cfg.dropout,
                          section="training.model")
        input_shape = cfg.model.input_shape

    elif model_path:
        # User model (h5 file)
        model = tf.keras.models.load_model(model_path)
        input_shape = tuple(model.input.shape[1:])

    elif cfg.resume_training_from:
        # Model saved during a previous training
        model = tf.keras.models.load_model(cfg.resume_training_from)
        input_shape = tuple(model.input.shape[1:])
    else:
        raise RuntimeError("\nInternal error: should have model,"
                           "model_path or resume_training_from")

    return model, input_shape

class PrintLREpoch(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        lr = self.model.optimizer.learning_rate

        # Handle schedules
        if isinstance(lr, tf.keras.optimizers.schedules.LearningRateSchedule):
            lr = lr(self.model.optimizer.iterations)

        lr = tf.keras.backend.get_value(lr)
        print(f"\n\nEpoch {epoch + 1}: lr = {lr:.6e}")

def get_callbacks(callbacks_dict: DictConfig, output_dir: str = None, logs_dir: str = None,
                  saved_models_dir: str = None) -> List[tf.keras.callbacks.Callback]:
    """
    This function creates the list of Keras callbacks to be passed to 
    the fit() function including:
      - the Model Zoo built-in callbacks that can't be redefined by the
        user (ModelCheckpoint, TensorBoard, CSVLogger).
      - the callbacks specified in the config file that can be either Keras
        callbacks or custom callbacks (learning rate schedulers).

    For each callback, the attributes and their values used in the config
    file are used to create a string that is the callback instantiation as
    it would be written in a Python script. Then, the string is evaluated.
    If the evaluation succeeds, the callback object is returned. If it fails,
    an error is thrown with a message saying that the name and/or arguments
    of the callback are incorrect.

    The user may use the ModelSaver callback to periodically save the model 
    at the end of an epoch. If it is not used, a default ModelSaver is added
    that saves the model at the end of each epoch. The model file is named
    last_model.keras and is saved in the output_dir/saved_models_dir directory
    (same directory as best_model.keras).

    The function also checks that there is only one learning rate scheduler
    in the list of callbacks.

    Args:
        callbacks_dict (DictConfig): dictionary containing the 'training.callbacks'
                                     section of the configuration.
        output_dir (str): directory root of the tree where the output files are saved.
        logs_dir (str): directory the TensorBoard logs and training metrics are saved.
        saved_models_dir (str): directory where the trained models are saved.

    Returns:
        List[tf.keras.callbacks.Callback]: A list of Keras callbacks to pass
                                           to the fit() function.
    """

    message = "\nPlease check the 'training.callbacks' section of your configuration file."
    lr_scheduler_names = lr_schedulers.get_scheduler_names()
    num_lr_schedulers = 0

    # Generate the callbacks used in the config file (there may be none)
    callback_list = []
    if callbacks_dict is not None:
        if type(callbacks_dict) != DefaultMunch:
            raise ValueError(f"\nInvalid callbacks syntax{message}")
        for name in callbacks_dict.keys():
            if name in ("ModelCheckpoint", "TensorBoard", "CSVLogger"):
                raise ValueError(f"\nThe `{name}` callback is built-in and can't be redefined.{message}")
            if name in lr_scheduler_names:
                text = f"lr_schedulers.{name}"
            else:
                text = f"tf.keras.callbacks.{name}"

            # Add the arguments to the callback string
            # and evaluate it to get the callback object
            text += collect_callback_args(name, args=callbacks_dict[name], message=message)
            try:
                callback = eval(text)
            except ValueError as error:
                raise ValueError(f"\nThe callback name `{name}` is unknown, or its arguments are incomplete "
                                 f"or invalid\nReceived: {text}{message}") from error
            callback_list.append(callback)

            if name in lr_scheduler_names + ["ReduceLROnPlateau", "LearningRateScheduler"]:
                num_lr_schedulers += 1

    # Check that there is only one scheduler
    if num_lr_schedulers > 1:
        raise ValueError(f"\nFound more than one learning rate scheduler{message}")

    # Add the Keras callback that saves the best model obtained so far
    callback = tf.keras.callbacks.ModelCheckpoint(
                        filepath=os.path.join(output_dir, saved_models_dir, "best_model.keras"),
                        save_best_only=True,
                        save_weights_only=False,
                        monitor="val_accuracy",
                        mode="max")
    callback_list.append(callback)

    # Add the Keras callback that saves the model at the end of the epoch
    callback = tf.keras.callbacks.ModelCheckpoint(
                    filepath=os.path.join(output_dir, saved_models_dir, "last_epoch_model.keras"),
                    save_best_only=False,
                    save_weights_only=False,
                    monitor="val_accuracy",
                    mode="max")
    callback_list.append(callback)

    # Add the TensorBoard callback
    callback = LRTensorBoard(log_dir=os.path.join(output_dir, logs_dir))
    callback_list.append(callback)

    # Add the CVSLogger callback (must be last in the list 
    # of callbacks to make sure it records the learning rate)
    callback = tf.keras.callbacks.CSVLogger(os.path.join(output_dir,
                                                         logs_dir, "metrics", "train_metrics.csv"))
    callback_list.append(callback)

    return callback_list

def log_training_metrics_mlflow(cfg, history, early_stop_cb):
    # Epoch actually stopped
    stopped_epoch = history.epoch[-1] + 1

    mlflow.log_metric("stopped_epoch", stopped_epoch)

    if early_stop_cb is not None:
        # Epoch with best monitored metric
        best_epoch = early_stop_cb.best_epoch  # 0-indexed

        # Extract training loss/accuracy for that epoch
        best_train_loss = history.history['loss'][best_epoch]
        acc_list = history.history.get('accuracy')
        if acc_list is None:
            best_train_acc = None
        else:
            best_train_acc = acc_list[best_epoch]

        mlflow.log_metric("best_epoch", best_epoch)
        mlflow.log_metric("best_loss", best_train_loss)
        mlflow.log_metric("best_accuracy", best_train_acc)


def get_tensorboard_profile_cb():
    logdir = "logs/profile"

    tb_cb = tf.keras.callbacks.TensorBoard(
        log_dir=logdir,
        histogram_freq=0,
        profile_batch=(100, 150)  # profile only these batches
    )

    return tb_cb

def check_tuner_cfg(cfg):
    tuner_cfg = cfg.keras_tuner
    if tuner_cfg is None:
        raise ValueError("\nPlease check the 'keras_tuner' section of your configuration file.")

    if tuner_cfg.max_trials is None:
        raise ValueError("\nPlease check the 'keras_tuner.max_trials' section of your configuration file.")

    if tuner_cfg.executions_per_trial is None:
        raise ValueError("\nPlease check the 'keras_tuner.executions_per_trial' section of your configuration file.")

def get_early_stopping_cb(cfg: DictConfig):
    if cfg.training.EarlyStopping is not None:
        return tf.keras.callbacks.EarlyStopping(
            monitor=cfg.training.EarlyStopping.monitor,
            mode=cfg.training.EarlyStopping.mode,
            patience=cfg.training.EarlyStopping.patience,
            restore_best_weights=cfg.training.EarlyStopping.restore_best_weights
        )

def get_keras_tuner_tensorboard_cb(logDir: str):
    tb_cb = tf.keras.callbacks.TensorBoard(
        log_dir=logDir,
    )

    return tb_cb

def train_keras_tuner(cfg: DictConfig,
                      train_ds: tf.data.Dataset,
                      valid_ds: tf.data.Dataset,
                      class_weights: Dict[int, float],
                      test_ds: Optional[tf.data.Dataset] = None,
                      callbacks: Optional[List[tf.keras.callbacks.Callback]] = None,
                      run_name: str = "keras_tuner_base") -> str:
    """
    Runs Keras Tuner using the provided configuration and datasets.

    Args:
        cfg (DictConfig): The entire configuration file dictionary.
        train_ds (tf.data.Dataset): training dataset loader.
        valid_ds (tf.data.Dataset): validation dataset loader.
        test_ds (Optional, tf.data.Dataset): test dataset dataset loader.
        callbacks (Optional, List[tf.keras.callbacks.Callback]): list of callbacks.
        run_name (str): name of the run

    Returns:
        Path to the best model obtained
    """
    check_tuner_cfg(cfg)

    output_dir = HydraConfig.get().runtime.output_dir

    class_names = cfg.dataset.class_names
    num_classes = len(class_names)

    early_stop_cb = get_early_stopping_cb(cfg)
    tb_cb = get_keras_tuner_tensorboard_cb(output_dir)

    if callbacks is None:
        callbacks = []

    if early_stop_cb is not None:
        print("[INFO] Adding early stopping callback")
        callbacks.append(early_stop_cb)
    if tb_cb is not None:
        print("[INFO] Adding tensorboard callback")
        callbacks.append(tb_cb)

    hypermodel = gmp_create_build_model(
        input_shape=cfg.training.model.input_shape,
        num_classes=num_classes,
        max_maccs=cfg.keras_tuner.max_maccs,
        max_num_params=cfg.keras_tuner.max_num_params,
    )

    tuner = keras_tuner.RandomSearch(
        hypermodel=hypermodel,
        objective=keras_tuner.Objective(
            "val_f1_macro",
            direction="max"
        ),
        max_trials=cfg.keras_tuner.max_trials,
        executions_per_trial=cfg.keras_tuner.executions_per_trial,
        directory=os.path.join(output_dir, "keras_tuner"),
        max_retries_per_trial=3,
        max_consecutive_failed_trials=20,
        project_name="HAR_GMP_TUNER",
        seed=get_random_seed(cfg),
    )

    tuner.search_space_summary()

    tuner.search(train_ds, validation_data=valid_ds,
                 epochs=cfg.training.epochs, callbacks=callbacks,
                 class_weight=class_weights)

    best_model = tuner.get_best_models(num_models=1)[0]

    # Show best model summary
    print("\nBest model summary:")
    best_model.summary()

    # Save the best model
    best_model_path = os.path.join(output_dir, "keras_tuner", "best_model.keras")
    best_model.save(best_model_path)

    return best_model_path


def train(cfg: DictConfig = None, train_ds: tf.data.Dataset = None,
          valid_ds: tf.data.Dataset = None,
          test_ds: Optional[tf.data.Dataset] = None,
          callbacks: Optional[List[tf.keras.callbacks.Callback]] = None,
          run_name: str = "base") -> str:
    """
    Trains the model using the provided configuration and datasets.

    Args:
        cfg (DictConfig): The entire configuration file dictionary.
        train_ds (tf.data.Dataset): training dataset loader.
        valid_ds (tf.data.Dataset): validation dataset loader.
        test_ds (Optional, tf.data.Dataset): test dataset dataset loader.
        callbacks (Optional, List[tf.keras.callbacks.Callback]): list of callbacks.
        run_name (str): name of the run

    Returns:
        Path to the best model obtained
    """

    #output_dir = cfg.output_dir
    output_dir = HydraConfig.get().runtime.output_dir

    # Make a separate directory for each run
    output_dir = os.path.join(output_dir, "run_" + str(run_name))

    os.makedirs(output_dir, exist_ok=True)

    saved_models_dir = cfg.general.saved_models_dir
    class_names = cfg.dataset.class_names
    num_classes = len(class_names)


    model, _ = load_model_to_train(cfg.training, model_path=cfg.general.model_path,
                                   num_classes=num_classes)

    # Info messages about the model that was loaded
    if cfg.training.model:
        cfm = cfg.training.model
        print(f"[INFO] : Using `{cfm.name}` model")
        log_to_file(cfg.output_dir, (f"Model name : {cfm.name}"))
        if cfm.pretrained_weights:
            print(f"[INFO] : Initialized model with '{cfm.pretrained_weights}' pretrained weights")
            log_to_file(cfg.output_dir,(f"Pretrained weights : {cfm.pretrained_weights}"))
        elif cfm.pretrained_model_path:
            print(f"[INFO] : Initialized model with weights from model file {cfm.pretrained_model_path}")
            log_to_file(cfg.output_dir, (f"Weights from model file : {cfm.pretrained_model_path}"))
        else:
            print("[INFO] : No pretrained weights were loaded, training from randomly initialized weights.")
    elif cfg.training.resume_training_from:
        print(f"[INFO] : Resuming training from model file {cfg.training.resume_training_from}")
        log_to_file(cfg.output_dir, (f"Model file : {cfg.training.resume_training_from}"))
    elif cfg.general.model_path:
        print(f"[INFO] : Loaded model file {cfg.general.model_path}")
        log_to_file(cfg.output_dir ,(f"Model file : {cfg.general.model_path}"))
    if cfg.dataset.name:
        log_to_file(output_dir, f"Dataset : {cfg.dataset.name}")

    if not cfg.training.resume_training_from:
        # Set frozen layers
        if not cfg.training.frozen_layers or cfg.training.frozen_layers == "None":
            model.trainable = True
        else:
            set_frozen_layers(model, frozen_layers=cfg.training.frozen_layers)
        # Set dropout rate
        set_dropout_rate(model, dropout_rate=cfg.training.dropout)

    model.summary()

    if cfg.training.steps_per_execution is None:
        cfg.training.steps_per_execution = 1
    print(f"[INFO] : steps_per_execution = {cfg.training.steps_per_execution}")

    # Compile the augmented model
    model.compile(loss=get_loss(num_classes=num_classes),
                  metrics=['accuracy'],
                  optimizer=get_optimizer(cfg=cfg.training.optimizer),
                  steps_per_execution=cfg.training.steps_per_execution)

    user_config_callbacks = get_callbacks(callbacks_dict=cfg.training.callbacks,
                              output_dir=output_dir,
                              saved_models_dir=saved_models_dir,
                              logs_dir=cfg.general.logs_dir)

    if callbacks is None:
        callbacks = []

    callbacks += user_config_callbacks
    callbacks.append(PrintLREpoch())

    callbacks.append(get_tensorboard_profile_cb())

    early_stop_cb = next((cb for cb in callbacks if isinstance(cb, EarlyStopping)), None)

    # check if determinism can be enabled
    if cfg.general.deterministic_ops:
        sample_ds = train_ds.take(1)
        tf.config.experimental.enable_op_determinism()
        if not check_training_determinism(model, sample_ds):
            print("[WARNING]: Some operations cannot be run deterministically.\
                   Setting deterministic_ops to False.")
            tf.config.experimental.enable_op_determinism.__globals__["_pywrap_determinism"].enable(False)

    runtime_csv_path = os.path.join(output_dir,
                                    cfg.general.logs_dir,
                                    "metrics",
                                    "train_runtime.csv")
    if os.path.isfile(runtime_csv_path):
        os.remove(runtime_csv_path)

    tf.config.run_functions_eagerly(False)

    print("Config eager is : ", tf.config.functions_run_eagerly())

    # Train the model
    print("Starting training...")
    print("Training for {} epochs, steps per epoch : {}, batch size : {}".format(cfg.training.epochs, cfg.training.steps_per_epoch, cfg.training.batch_size))
    start_time = timer()
    history = model.fit(train_ds,
                        validation_data=valid_ds,
                        epochs=cfg.training.epochs,
                        steps_per_epoch=cfg.training.steps_per_epoch,
                        callbacks=callbacks,
                        verbose=1)
    end_time = timer()
    #save the last epoch history in the log file
    last_epoch=log_last_epoch_history(cfg, output_dir)
    fit_run_time = int(end_time - start_time)
    average_time_per_epoch = round(fit_run_time / (int(last_epoch) + 1),2)
    print("Training runtime: " + str(timedelta(seconds=fit_run_time)))
    log_to_file(cfg.output_dir, (f"Training runtime : {fit_run_time} s\n" + f"Average time per epoch : {average_time_per_epoch} s"))

    with open(runtime_csv_path, "w", encoding='utf-8') as f:
        f.write("epochs,runtime\n")
        f.write(f"{cfg.training.epochs},{fit_run_time}\n")

    # Visualize training curves
    vis_training_curves(history=history, output_dir=output_dir)
    best_model_path = os.path.join(output_dir,
                                   saved_models_dir,
                                   "best_model.keras")
    best_model = tf.keras.models.load_model(best_model_path)
    # Save a copy of the best model if requested
    if cfg.training.trained_model_path:
        best_model.save(cfg.training.trained_model_path)
        print(f"[INFO] : Saved trained model in file {cfg.training.trained_model_path}")

    log_training_metrics_mlflow(cfg, history, early_stop_cb)

    # Evaluate keras best model on the validation set
    evaluate_keras_model(model_path=best_model_path, eval_ds=valid_ds,
                      class_names=class_names, output_dir=output_dir, name_ds="validation_set")

    if test_ds:
        # Evaluate keras best model on the test set
        evaluate_keras_model(model_path=best_model_path, eval_ds=test_ds,
                          class_names=class_names, output_dir=output_dir, name_ds="test_set")

    return best_model_path
