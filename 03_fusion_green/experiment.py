"""Main CLI interface for fusion_green experiments."""
import argparse
import logging
import multiprocessing as mp
import os
import pathlib as pl
import shutil
from datetime import datetime
from os import makedirs, path
from subprocess import call

import accuracy as acc
import tensorflow as tf
from box import Box
from fusion_green import (
    compile_network,
    create_fusion_net,
    infer_network,
    train_and_evaluate_sklearn,
    train_network,
)
from green_dataset import GreenDatasetPrecursor
from predictions import predict_model_reg
from rich import traceback
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression


def _copy_experiment(args):
    makedirs(args.copy_target)
    shutil.copyfile(f"{args.directory}/config.yml", f"{args.copy_target}/config.yml")

    # optional: edit file
    edit_file = input("Edit file? [y]/n:")
    if edit_file == "y" or edit_file == "":
        editor = os.environ.get("EDITOR", "vim")
        call([editor, f"{args.copy_target}/config.yml"])

    run_copied_experiment = input("Run the copied experiment? [y]/n:")
    return run_copied_experiment == "y" or run_copied_experiment == ""


def _expand_directory(directory):
    return path.abspath(path.expanduser(directory))


def _setup_parser(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--copy_target",
        type=_expand_directory,
        help="Copy config.yml of previous experiment and to target directory.",
    )
    parser.add_argument(
        "-d", "--debug", action="store_true", help="Increase logging level to DEBUG"
    )
    parser.add_argument(
        "-n",
        "--new-experiment",
        action="store_true",
        help="Copy all necessary files to a new directory.",
    )
    parser.add_argument("directory", type=_expand_directory, help="Experiment directory.")

    if args:
        return parser.parse_args(args)

    return parser.parse_args()


def _setup_new_experiment(args):
    makedirs(args.directory)
    shutil.copyfile("./config.yml.example", f"{args.directory}/config.yml")

    # optional: edit config file
    edit_file = input("Edit file? [y]/n:")
    if edit_file == "y" or edit_file == "":
        editor = os.environ.get("EDITOR", "vim")
        call([editor, f"{args.directory}/config.yml"])

    # optional: edit config file
    run_new_experiment = input("Run new experiment? [y]/n:")
    return run_new_experiment == "y" or run_new_experiment == ""


def _setup_logger(args):
    log_level = logging.DEBUG if args.debug else logging.INFO
    # file handler to log file
    fh = logging.FileHandler(f"{args.directory}/run.log", mode="a")
    # stream handler STDOUT
    sh = logging.StreamHandler()
    logging.basicConfig(
        handlers=[fh, sh],
        level=log_level,
        format="%(levelname)8s:%(name)8s: %(message)s",
    )

    # block some modules from spamming logs
    logger_blocklist = [
        "fiona",
        "rasterio",
        "matplotlib",
        "PIL",
        "h5py",
        "tensorboard",
        "MARKDOWN",
    ]
    for module in logger_blocklist:
        logging.getLogger(module).setLevel(logging.WARNING)

    logging.info("################################")
    logging.info(f"#### START {datetime.now():%Y-%m-%d %H:%M} ####")
    logging.info("################################")


def _load_data(extracts_data, **kwargs):
    prec = GreenDatasetPrecursor(extracts_data, **kwargs)
    breakpoint()

    ds_train, ds_val = prec.split_datasets_train_valid()
    ds_test_imbal, ds_test_bal = prec.prepare_test_bal_imbal_dataset()
    ds_test_dict = {"balanced": ds_test_bal, "imbalanced": ds_test_imbal}

    return ds_train, ds_val, ds_test_dict


def _setup_seeds(config):
    logging.debug(f"Setting Seeds to {config.reproducibility.seed}")
    # set base-python, numpy and tf seed
    tf.keras.utils.set_random_seed(config.reproducibility.seed)
    # might be obsolete, but legacy
    os.environ["PYTHONHASHSEED"] = str(config.reproducibility.seed)

    if config.reproducibility.global_determinism:
        # Enable 100% reproducibility on operations related to tensor and randomness.
        # https://www.tensorflow.org/api_docs/python/tf/config/experimental/enable_op_determinism
        logging.warning("set_global_determinism=True,setting full determinism, WILL BE SLOW!!")
        tf.config.experimental.enable_op_determinism()


def main() -> int:
    """Start the experiment."""
    # ~~~~~~~~~~ Preamble ~~~~~~~~~~~~~#
    # set up parser
    args = _setup_parser()

    if args.new_experiment:
        try:
            run_new_experiment = _setup_new_experiment(args)
            if not run_new_experiment:
                return 0
        except FileExistsError:
            logging.critical(f"ERROR: {args.directory} already exists!")
            return 1

    if args.copy_target:
        run_copied_experiment = _copy_experiment(args)
        if not run_copied_experiment:
            return 0
        else:
            # replace the directory and continue running
            args.directory = args.copy_target

    # parse config
    try:
        with open(f"{args.directory}/config.yml") as f:
            config = Box.from_yaml(f)
    except FileNotFoundError:
        logging.critical(f"No config.yml found in {args.directory}")
        return 1

    # rich traceback
    if args.debug:
        traceback.install(show_locals=True)

    # set up logger
    _setup_logger(args)
    logging.debug(f"Registered Arguments: {args!s}")
    logging.debug(f"Registered Config: {config!s}")

    # set seeds for reprodicibility
    _setup_seeds(config)

    log_dir = path.abspath(f"./logs/{args.directory.split('/')[-1]}")

    # ~~~~~~~~~~ Main Experiments ~~~~~#
    # read data
    logging.debug("Reading data")
    ds_train, ds_val, ds_test_dict = _load_data(
        config.data.extracts_data,
        experiment_root=pl.Path(args.directory),
        **config.precursor,
    )

    def dl_experiments(
        network_mode,
        load_paths: list[str] = [],
        classification=config.precursor.classification,
        n_classes=config.precursor.n_green_classes,
    ):
        # create the network
        model = create_fusion_net(
            ds_train,
            mode=network_mode,
            dropout_rate=config.model.dropout_rate,
            cnn_model=config.model.cnn_model,
            classification=classification,
            cnn_dense_layers=config.model.cnn_dense_layers,
            fusion_cnn_neurons=config.model.fusion_cnn_neurons,
            fusion_ann_neurons=config.model.fusion_ann_neurons,
            ann_hidden_layers=config.model.ann_hidden_layers,
            ann_hidden_layer_nodes=config.model.ann_hidden_layer_nodes,
            ann_bn_or_do=config.model.ann_bn_or_do,
            fusion_hidden_layers=config.model.fusion_hidden_layers,
            fusion_hidden_layer_nodes=config.model.fusion_hidden_layer_nodes,
            greennet_skipconn=config.model.greennet_skipconn,
        )
        # compile the fusion network setting optimization, loss, etc
        compiled_model = compile_network(
            model,
            config.model.initial_learning_rate,
            classification=classification,
            n_classes=n_classes,
        )

        compiled_model.save(f"{args.directory}/{config.data.model_prefix}_{network_mode}_full.h5")

        for p in load_paths:
            compiled_model.load_weights(p, by_name=True)

        # run the network for EPOCHS
        model_path = f"{args.directory}/{config.data.model_prefix}_{network_mode}.h5"
        _ = train_network(
            compiled_model=compiled_model,
            model_path=model_path,
            dataset_train=ds_train,
            dataset_val=ds_val,
            epochs=config.model.epochs,
            warmup_epochs=config.model.warmup_epochs,
            warmup_lr=config.model.warmup_lr,
            reduce_on_plateau_patience=config.model.reduce_on_plateau_patience,
            early_stopping_patience=config.model.early_stopping_patience,
            log_dir=f"{log_dir}/{network_mode}",
        )
        # # assess accuracy end2end
        acc.assess_model_accuracy(
            compiled_model,
            model_path,
            network_mode,
            ds_test_dict,
            args.directory,
            classification,
        )
        infer_network(compiled_model, model_path, ds_test_dict["imbalanced"], network_mode)

        return model_path

    load_paths = []

    if config.experiments.cnn:
        cnn_path = dl_experiments("cnn")
        load_paths.append(cnn_path)

    if config.experiments.ann:
        # deactivate augmentation to speed things up
        if config.precursor.augment_training_ds:
            ds_train.toggle_augment()
        if config.precursor.augment_valid_ds:
            ds_val.toggle_augment()

        ann_path = dl_experiments("ann")
        load_paths.append(ann_path)

        # activate augmentation again
        if config.precursor.augment_training_ds:
            ds_train.toggle_augment()
        if config.precursor.augment_valid_ds:
            ds_val.toggle_augment()

    if config.experiments.fusion:
        dl_experiments("fusion", load_paths)

    if config.experiments.rf:
        if config.precursor.classification:
            train_and_evaluate_sklearn(
                RandomForestClassifier(
                    n_estimators=500,
                    random_state=config.reproducibility.seed,
                    n_jobs=mp.cpu_count(),
                ),
                ds_train,
                ds_test_dict,
                experiment_title="rf",
                experiment_path=args.directory,
                classification=True,
            )
        else:
            train_and_evaluate_sklearn(
                RandomForestRegressor(
                    n_estimators=500,
                    random_state=config.reproducibility.seed,
                    n_jobs=mp.cpu_count(),
                ),
                ds_train,
                ds_test_dict,
                experiment_title="rf",
                experiment_path=args.directory,
                classification=False,
            )

    if config.experiments.lm:
        if not config.precursor.classification:
            train_and_evaluate_sklearn(
                LinearRegression(),
                ds_train,
                ds_test_dict,
                experiment_title="lm",
                experiment_path=args.directory,
                classification=False,
            )
        else:
            logging.critical("Choosing precursor.classification conflicts with experiments.lm!")

    # write outputs
    output_file_prefix = f"{args.directory}/pred_imbal"

    if config.output.gpkg:
        ds_test_dict["imbalanced"].write_gpkg(
            filename=f"{output_file_prefix}.gpkg",
            reference_raster_file=config.data.reference_raster,
            tile_size=ds_train.img.shape[1:3],
        )
    if config.output.h5:
        ds_test_dict["imbalanced"].write_h5(f"{output_file_prefix}.h5")

    # remove unnecessary datasets to reduce memory footprint prior to predictions
    del ds_train, ds_val, ds_test_dict

    if config.output.prediction.extracts_data and not config.precursor.classification:
        models = [k for k, v in config.experiments.items() if v]
        predict_model_reg(
            pl.Path(config.output.prediction.extracts_data),
            pl.Path(args.directory),
            models,
        )

    logging.info("################################")
    logging.info(f"#### END {datetime.now():%Y-%m-%d %H:%M}   ####")
    logging.info("################################")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
