"""Predict trained models on entire dataset."""
import pathlib as pl
import sys

import box
import h5py
import joblib as jl
import numpy as np
import tensorflow as tf
from fusion_green import r2
from green_dataset import GreenDaskDataset

BATCH_SIZE = 256


def predict_model_reg(h5_data_path: pl.Path, experiment_path: pl.Path, models: list[str]) -> None:
    """Predict models to full dataset."""
    scaling_factors = experiment_path / "scaling_factors.pickle"
    dataset = GreenDaskDataset(h5_data_path, scaling_factors, batch_size=BATCH_SIZE)

    for model in models:
        print(f"Predicting {model}")
        if model in ["ann", "cnn", "fusion"]:
            predictions = _predict_model_reg_tf(dataset, experiment_path, model)
        elif model in ["lm", "rf"]:
            predictions = _predict_model_reg_skl(dataset, experiment_path, model)
        else:
            raise NotImplementedError
        del dataset

        _write_array_to_file(predictions, experiment_path, model)

    dataset = GreenDaskDataset(h5_data_path, scaling_factors)  # , batch_size=BATCH_SIZE)
    # add ulc and urb information for being able to spatialize the h5 predictions
    _write_array_to_file(dataset.ulc.compute(), experiment_path, "ulc")
    _write_array_to_file(dataset.eua.compute(), experiment_path, "eua")
    _write_array_to_file(dataset.urb.compute(), experiment_path, "urb")
    del dataset


def _predict_model_reg_tf(
    dataset: GreenDaskDataset, experiment_path: pl.Path, model: str
) -> np.array:
    # load the model arcitecture
    cust_obj = {"r2": r2}
    tf_model = tf.keras.models.load_model(
        experiment_path / f"model_{model}_full.h5", custom_objects=cust_obj
    )
    # load weights achieved by best training run (as measured by val-loss)
    tf_model.load_weights(experiment_path / f"model_{model}.h5")
    predictions = tf_model.predict(
        dataset, workers=7, use_multiprocessing=True, verbose=1  # , max_queue_size=6
    )
    return predictions


def _predict_model_reg_skl(
    dataset: GreenDaskDataset, experiment_path: pl.Path, model: str
) -> np.array:
    experiment_path = pl.Path(args.directory)
    model = "lm"
    # load the trained model
    skl_model = jl.load(experiment_path / f"model_{model}.pkl")
    predictions = skl_model.predict(dataset.osm)
    return predictions


def _write_array_to_file(arr: np.ndarray, experiment_path: pl.Path, model: str) -> None:
    with h5py.File(experiment_path / "full_predictions.h5", "a") as f:
        f.create_dataset(name=model, data=arr, shape=arr.shape, dtype=arr.dtype)


if __name__ == "__main__":
    experiment_root = pl.Path(sys.argv[1])

    with open(experiment_root / "config.yml") as f:
        config = box.Box.from_yaml(f)

    models = [k for k, v in config.experiments.items() if v]

    predict_model_reg(config.output.prediction.extracts_data, experiment_root, models)
