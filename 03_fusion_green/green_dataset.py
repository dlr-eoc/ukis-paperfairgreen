"""Provides datasets for fusion green experiments."""
import hashlib
import logging
import pathlib as pl
import pickle
import shelve
import types
from itertools import chain

import dask.array as da
import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio as rio
from h5py import File
from numba import njit
from shapely.geometry import box
from tensorflow.keras.utils import Sequence


class GreenDatasetPrecursor:
    def __init__(
        self,
        filename,
        n_obs=None,
        eua_completely_covered=True,
        eua_scaling="linear",
        max_dist_urban_proximity=None,
        sampling_scheme="simple",
        osm_scaling="linear",
        train_val_test_split=[0.6, 0.2, 0.2],
        batch_size=256,
        shuffle=True,
        balance_training=True,
        balance_validation=True,
        balance_weight_ratio_green_training=1,
        balance_weight_ratio_green_validation=1,
        clean_on_read=True,
        classification=False,
        n_green_classes=5,
        augment_training_ds=False,
        augment_valid_ds=False,
        experiment_root: pl.Path = None,
    ):
        """Set up a dataset precursor sorting, pruning and splitting green extracts for TF."""
        # init using parameters
        self.filename = filename
        self.n_obs = n_obs
        self.eua_completely_covered = eua_completely_covered
        self.max_dist_urban_proximity = max_dist_urban_proximity
        self.train_val_test_split = train_val_test_split
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.balance_training = balance_training
        self.balance_validation = balance_validation
        self.balance_weight_ratio_green_training = balance_weight_ratio_green_training
        self.balance_weight_ratio_green_validation = balance_weight_ratio_green_validation
        self.clean_on_read = clean_on_read
        self.classification = classification
        self.n_green_classes = n_green_classes
        self.augment_train_ds = augment_training_ds
        self.augment_valid_ds = augment_valid_ds
        self.experiment_root = experiment_root

        # init using defaults
        self.network_modes = ["fusion", "cnn", "ann"]
        self.sampling_schemes = ["simple", "random"]

        # index variables, used for shuffeling, prediction, etc.
        self.all_idx = self.train_idx = self.valid_idx = self.test_idx = None

        # variables to store the data
        self.img = self.osm = self.eua = self.urb = self.ulc = None
        # load data
        self._read_h5()

        if self.classification:
            self._convert_target_to_classification()

        self.set_sampling_scheme(sampling_scheme)

        # shuffle if set
        if self.shuffle:
            self._shuffle_idx()

        # balance if set
        if self.balance_training:
            self.train_idx = self._balance_idx(
                self.train_idx, self.balance_weight_ratio_green_training, "training"
            )
        if self.balance_validation:
            self.valid_idx = self._balance_idx(
                self.valid_idx, self.balance_weight_ratio_green_validation, "validation"
            )

    def _apply_sampling_scheme(self):
        if np.round(sum(self.train_val_test_split), 3) > 1:
            raise ValueError(f"{np.sum(self.train_val_test_split)=} > 1!")

        last_train_idx = int(len(self.all_idx) * self.train_val_test_split[0])
        last_valid_idx = last_train_idx + int(len(self.all_idx) * self.train_val_test_split[1])

        if self.sampling_scheme == "simple":
            logging.debug("Apply 'simple' sampling scheme.")

        elif self.sampling_scheme == "random":
            logging.warning("Apply 'random' sampling scheme. Be careful with overlapping extracts.")
            # same as simple, but shffle all_idx
            np.random.shuffle(self.all_idx)

        else:
            raise NotImplementedError(
                f"No method for applying sampling scheme {self.sampling_scheme}."
            )

        self.train_idx = self.all_idx[:last_train_idx]
        self.valid_idx = self.all_idx[last_train_idx:last_valid_idx]
        self.test_idx = self.all_idx[last_valid_idx:]

    def _apply_scaling(self, osm_scaling="linear", eua_scaling="linear"):
        logging.debug("Apply scaling to img.")
        img_min = self.img.min(axis=(0, 1, 2), keepdims=True)
        img_max = self.img.max(axis=(0, 1, 2), keepdims=True)
        self.img = np.divide((self.img - img_min), (img_max - img_min), dtype=np.float32)

        logging.debug("Apply scaling to eua.")
        eua_min = self.eua.min(axis=0, keepdims=True)
        eua_max = self.eua.max(axis=0, keepdims=True)
        self.eua = np.divide((self.eua - eua_min), (eua_max - eua_min), dtype=np.float32)
        if eua_scaling == "sqrt":
            logging.info("Apply sqrt scaling to label.")
            self.eua = np.sqrt(self.eua)

        np.seterr(divide="ignore")

        def minmax_scale_osm():
            osm_min = self.osm.min(axis=0, keepdims=True)
            osm_max = self.osm.max(axis=0, keepdims=True)
            self.osm = np.nan_to_num(
                np.divide((self.osm - osm_min), (osm_max - osm_min), dtype=np.float32)
            )
            return osm_min, osm_max

        if osm_scaling == "linear":
            logging.debug("Apply 'linear' scaling to osm.")
            osm_min, osm_max = minmax_scale_osm()
        elif osm_scaling == "log":
            logging.debug("Apply 'log' scaling to osm.")
            self.osm[self.osm < 1] = 1
            self.osm = np.log(self.osm)
            osm_min, osm_max = minmax_scale_osm()
        elif osm_scaling == "sqrt":
            logging.debug("Apply 'sqrt' scaling to osm.")
            self.osm = np.sqrt(self.osm)
            osm_min, osm_max = minmax_scale_osm()
        else:
            raise NotImplementedError(f"No method for applying osm scaling '{osm_scaling}'")

        # some assertions just to be sure
        assert np.max(self.img) == 1.0, f"{np.max(self.img)=} > 1!"
        assert np.min(self.img) == 0.0, f"{np.min(self.img)=} < 0!"
        assert np.max(self.eua) == 1.0, f"{np.max(self.eua)=} > 1!"
        assert np.min(self.eua) == 0.0, f"{np.min(self.eua)=} < 0!"
        assert np.max(self.osm) == 1.0, f"{np.max(self.osm)=} > 1!"
        assert np.min(self.osm) == 0.0, f"{np.min(self.osm)=} < 0!"

        scaling_factors = {
            "img_min": img_min,
            "img_max": img_max,
            "eua_min": eua_min,
            "eua_max": eua_max,
            "osm_min": osm_min,
            "osm_max": osm_max,
        }
        scaling_stats_file = self.experiment_root / "scaling_factors.pickle"

        with open(scaling_stats_file, "wb") as f:
            pickle.dump(scaling_factors, f)

    def _shuffle_idx(self):
        np.random.shuffle(self.train_idx)
        np.random.shuffle(self.valid_idx)
        np.random.shuffle(self.test_idx)

    def _balance_idx(self, idx, ratio, index_name):
        """Method to balance training and validation data, i.e. over/under sampling greens."""
        if not self.classification:
            # derive which label is max by checking whether the first is larger than half of max
            eua_max = np.max(self.eua)

            # green (1) is max when self.eua[,0] smaller than haf of eua_max
            eua_which_max = (self.eua[idx, 0] <= (eua_max / 2)).astype(int)

            # determine number of observations per eua class
            n_other = np.sum(eua_which_max == 0)
            n_green = np.sum(eua_which_max == 1)
            # determine optimal number of green tiles when applying the balancing (over/under
            # sampling)
            optimal_number_of_green = int(n_other * ratio)
            logging.debug(f"{n_other=}, {n_green=}, {optimal_number_of_green=}")

            if n_green > optimal_number_of_green:
                # if number of majority green tiles exceeds needed number, reduce greens
                logging.debug(
                    "Num. of maj. green tiles is LOWER than number of optimal green tiles."
                )
                n_green = optimal_number_of_green
                idx_other = idx[np.where(eua_which_max == 0)][:n_other]
                idx_green = idx[np.where(eua_which_max == 1)][:n_green]
            else:
                # if number of majority green tiles is too low, reduce maj. other tiles
                logging.debug(
                    "Num. of maj. green tiles is HIGHER than number of optimal green tiles."
                )
                n_other = int(n_green / ratio)
                idx_other = idx[np.where(eua_which_max == 0)][:n_other]
                idx_green = idx[np.where(eua_which_max == 1)][:n_green]

            # reassemble idx
            idx = np.concatenate((idx_other, idx_green))
            logging.info(f"Balancing {index_name} dataset: Reduced to {n_other=}, {n_green=}.")

        else:
            # how many observations per class
            class_counts = np.sum(self.eua[idx], axis=0)

            # what is the maximum count of observations applying the oversampling rate
            max_n = np.min(class_counts) / ratio

            # find number of observations per class as minimum of max_n and the class count
            obs_per_class = np.min(
                (class_counts, np.array([max_n] * self.n_green_classes)), axis=0
            ).astype(int)

            # argwhere returns a 2d array ax0 = row, ax1 = the class
            class_idx = np.argwhere(self.eua)[idx]

            if self.shuffle:
                np.random.shuffle(class_idx)

            idx = []
            for i in range(self.n_green_classes):
                # split class_idx into classes ...
                idx_tmp = class_idx[class_idx[:, 1] == i, 0]
                # ... and trim to correct size
                idx_tmp = idx_tmp[: obs_per_class[i]]

                idx.extend(idx_tmp)

            logging.info(f"Balancing {index_name} dataset: Reduced to {obs_per_class=}")

        # some info on very small datasets
        lens = len(idx)
        if lens == 0:
            logging.critical(
                "The chosen precursor parameters lead to a {index_name} dataset with 0"
                "records. Stopping."
            )
            raise SystemExit(1)

        elif lens < 1000:
            logging.warning(f"{lens} is a very small number of {index_name} records!")

        return idx

    def _balance_test_idx(self):
        """Method to balance test data. Returns balanced index"""
        label_majority_indices = [[] for _ in range(self.eua.shape[1])]

        for idx in self.test_idx:
            eua_tmp = self.eua[idx, :]
            maj_idx = int(np.where(eua_tmp == np.amax(eua_tmp))[0])
            label_majority_indices[maj_idx].append(idx)

        lens = [len(x) for x in label_majority_indices]
        logging.info(
            f"Balancing test dataset: Reduced to {self.eua.shape[1]} x {min(lens)} features"
        )

        if min(lens) < 1000:
            logging.warning(
                f"{min(lens)} x {self.eua.shape[1]} is a very small number of validation records "
                "for the balanced and will prohibit meaningful results!"
            )

        # zip to combine and trim to min len, a.k.a. balance
        # chain to flatten
        return np.fromiter(chain(*zip(*label_majority_indices)), dtype=self.all_idx.dtype)

    def _convert_target_to_classification(self):
        # numeric conversion of continuous variable to classes
        classes = np.floor(self.eua[:, 1] * self.n_green_classes)
        # the last bin classes == self.n_green_classes only consists of cases where self.eua[:, 1]
        # is EXACTLY 1. This case is added to the bin below
        classes = np.where(classes == self.n_green_classes, self.n_green_classes - 1, classes)

        # one-hot-encoding https://stackoverflow.com/a/42874726/3250126
        self.eua = np.eye(self.n_green_classes)[classes.astype(int)]

    def _filter_observations(self, keep_idx, drop_entirely=False):
        """Filter all data sets by an index vector"""
        self.all_idx = self.all_idx[keep_idx]

        if drop_entirely:
            self.img = self.img[self.all_idx]
            self.osm = self.osm[self.all_idx]
            self.eua = self.eua[self.all_idx]
            self.urb = self.urb[self.all_idx]
            self.ulc = self.ulc[self.all_idx]
            self._reset_idx()

        if self.train_idx is not None:
            self.train_idx = [x for x in self.train_idx if x in self.all_idx]
        if self.valid_idx is not None:
            self.valid_idx = [x for x in self.valid_idx if x in self.all_idx]
        if self.test_idx is not None:
            self.test_idx = [x for x in self.test_idx if x in self.all_idx]

    def _read_h5(self):
        """Read data from h5 files."""
        with File(self.filename, "r") as f:
            # first read eua and find out which are fully covered (limited to max self.n_obs)
            logging.debug("Reading eua values to set up the index")
            eua_tmp = f["eua"][: self.n_obs]
            # set up index used to pre-select which elements are going to be read
            total_n_obs = len(eua_tmp)
            read_idx = np.arange(total_n_obs)
            # limit index by eua coverage
            if self.eua_completely_covered:
                logging.debug("Checking eua values for full coverage")
                max_covg = np.max(np.sum(eua_tmp, axis=1))
                read_idx = read_idx[np.sum(eua_tmp, axis=1) > (max_covg * 0.98)]
            del eua_tmp

            # (further) limit data by urban proximity
            if self.max_dist_urban_proximity:
                # read urban distance aggregates
                logging.debug("Reading urb values.")
                urb_tmp = f["urb"][: self.n_obs]
                urb_tmp = urb_tmp[read_idx]
                logging.debug("Checking urb values for max_dist_urban_proximity")
                read_idx = read_idx[urb_tmp[:, 3] <= self.max_dist_urban_proximity]
                del urb_tmp

            read_idx.sort()

            logging.info(f"Experiment uses {len(read_idx):,} observations.")

            if len(read_idx) == 0:
                logging.critical(
                    "The chosen value for `max_dist_urban_proximity` results in "
                    "zero observations"
                )
                raise SystemExit(1)

            if len(read_idx) < 1000:
                logging.warning(
                    "The chosen parameter set results in very few observations"
                    f" (N = {len(read_idx)}). This might lead to unexpected outcomes!"
                )

            # create hash for caching
            read_idx_hash = hashlib.md5()
            for x in np.sort(read_idx):
                read_idx_hash.update(str(x).encode("UTF-8"))

            cache_dir = pl.Path("./cache/")
            cache_dir.mkdir(exist_ok=True)
            cache_file = pl.Path(f"{cache_dir}/{read_idx_hash.hexdigest()}.cache")

            if pl.Path(f"{cache_file}.dat").exists():
                logging.info(f"Loading cached data from {cache_file}")
                with shelve.open(str(cache_file)) as cf:
                    self.eua = cf["eua"]
                    self.img = cf["img"]
                    self.osm = cf["osm"]
                    self.urb = cf["urb"]
                    self.ulc = cf["ulc"]

            else:
                # estimate the size of the img numpy arrays in memory
                img_size_bytes = int(
                    len(read_idx)
                    * (
                        np.product(f["img"].shape[-3:])
                        + np.product(f["osm"].shape[-1:])
                        + np.product(f["eua"].shape[-1:])
                        + np.product(f["urb"].shape[-1:])
                        + np.product(f["ulc"].shape[-1:])
                    )
                    * 4  # 4 bytes = 32 bit
                )
                logging.debug(
                    f"expected byte size of img numpy arrays: {img_size_bytes:,} bytes / "
                    f"{img_size_bytes / 1024**3: .2f} GBytes."
                )

                # read all data heavy part of the data
                logging.debug("Reading all eua data.")
                eua = f["eua"][: self.n_obs]
                logging.debug("Subsetting valid eua data.")
                self.eua = eua[read_idx]
                del eua

                logging.debug("Read image using dask")
                f_img = f["img"]
                # reading h5 with dask from random indexes is 5 times faster than with plain h5py
                f_img_da = da.from_array(f_img, chunks="auto")
                self.img = f_img_da[read_idx].compute()
                logging.debug(f"{self.img.shape=}")

                logging.debug("Reading all osm data.")
                osm = f["osm"][: self.n_obs]
                logging.debug("Subsetting valid osm data and converting.")
                self.osm = osm[read_idx]
                del osm

                logging.debug("Reading all ulc data.")
                ulc = f["ulc"][: self.n_obs]
                logging.debug("Subsetting valid ulc data.")
                self.ulc = ulc[read_idx]
                del ulc

                logging.debug("Reading all urb data.")
                urb = f["urb"][: self.n_obs]
                logging.debug("Subsetting valid urb data.")
                self.urb = urb[read_idx]
                del urb

                # save to cache
                with shelve.open(str(cache_file)) as cf:
                    logging.info(f"Saving data to cache as {cache_file}")
                    cf["eua"] = self.eua
                    cf["img"] = self.img
                    cf["osm"] = self.osm
                    cf["urb"] = self.urb
                    cf["ulc"] = self.ulc

        # set up self.all_idx which is used later to apply sampling etc.
        self._reset_idx()

        # some groups might only only have 0 values and can therefore be dropped
        if self.clean_on_read:
            self._remove_osm_zero_only_columns()
            self._remove_osm_zero_only_rows()

    def _remove_osm_zero_only_columns(self):
        """Remove columns for which the osm data only contains zeros, such as for vineyards in
        non-wine-producing regions."""
        logging.debug(f"Removing {np.sum(np.max(self.osm, axis=0) == 0)} osm zero only columns.")
        self.osm = self.osm[:, np.max(self.osm, axis=0) != 0]

    def _remove_osm_zero_only_rows(self):
        """Remove rows which only contain 0 for all OSM features."""
        idx_to_keep = np.count_nonzero(self.osm, axis=1) != 0
        logging.debug(
            f"Removing {len(idx_to_keep) - np.sum(idx_to_keep)}/{len(idx_to_keep)} "
            "osm zero only rows."
        )
        self._filter_observations(idx_to_keep, drop_entirely=True)

    def _reset_idx(self):
        self.all_idx = np.arange(self.ulc.shape[0])

    def limit_by_urban_proximity(self, distance, method="leq_max", drop_entirely=False):
        # reset previous filters, sampling schemes etc.
        self._reset_idx()

        # filter methods
        if distance:
            if method == "leq_max":
                idx_tmp = self.urb[:, 3] <= distance
                self._filter_observations(idx_tmp, drop_entirely)
            else:
                raise NotImplementedError
        # reapply sampling scheme on new set of indexes
        self._apply_sampling_scheme()

    def prune_osm_columns_by_nonzero_proportion(self, min_p):
        """Prune osm variables based on minimum non-null cells by proportion."""
        osm_col_nonzero_proportion = np.count_nonzero(self.osm, axis=0) / self.osm.shape[0]
        idx_to_keep = osm_col_nonzero_proportion >= min_p
        self.osm = self.osm[:, idx_to_keep]
        logging.info(
            f"Pruned {np.sum(np.invert(idx_to_keep))} columns from osm with p < {min_p} "
            "nonzero observations"
        )

    def prune_osm_columns_by_nonzero_number(self, min_n):
        """Prune osm variables based on minimum non-null cells by number."""
        osm_col_nonzero_n = np.count_nonzero(self.osm, axis=0)
        idx_to_keep = osm_col_nonzero_n >= min_n
        self.osm = self.osm[:, idx_to_keep]
        logging.info(
            f"Pruned {np.sum(np.invert(idx_to_keep))} columns from osm with n < {min_n} "
            "nonzero observations"
        )

    def set_sampling_scheme(self, scheme):
        if scheme.lower() in self.sampling_schemes:
            self.sampling_scheme = scheme.lower()
        else:
            raise NotImplementedError(
                f"Sampling scheme {scheme} is not implemented. Choose one of: "
                f"{', '.join(self.sampling_schemes)}"
            )
        self._apply_sampling_scheme()

    def split_datasets_train_valid(self):
        """Export Datasets for training and validation of the network"""
        logging.debug("Creating training dataset.")
        ds_train = GreenDataset(
            img=self.img[self.train_idx],
            osm=self.osm[self.train_idx],
            eua=self.eua[self.train_idx],
            shuffle=self.shuffle,
            batch_size=self.batch_size,
            classification=self.classification,
            augment=self.augment_train_ds,
        )
        logging.debug("Creating validation dataset.")
        ds_valid = GreenDataset(
            img=self.img[self.valid_idx],
            osm=self.osm[self.valid_idx],
            eua=self.eua[self.valid_idx],
            shuffle=False,
            batch_size=self.batch_size,
            classification=self.classification,
            augment=self.augment_valid_ds,
        )
        return ds_train, ds_valid

    def prepare_test_bal_imbal_dataset(self):
        """Export one imbalanced (unaltered) and one balanced test dataset."""
        logging.debug("Creating imbalanced training dataset.")
        ds_test_imbalanced = GreenDataset(
            img=self.img[self.test_idx],
            osm=self.osm[self.test_idx],
            eua=self.eua[self.test_idx],
            urb=self.urb[self.test_idx, 3],
            ulc=self.ulc[self.test_idx],
            shuffle=False,
            batch_size=self.batch_size,
            classification=self.classification,
        )
        logging.debug("Deriving balanced training dataset index.")
        balanced_idx = self._balance_test_idx()
        logging.debug("Creating balanced training dataset.")
        ds_test_balanced = GreenDataset(
            img=self.img[balanced_idx],
            osm=self.osm[balanced_idx],
            eua=self.eua[balanced_idx],
            urb=self.urb[balanced_idx, 3],
            ulc=self.ulc[balanced_idx],
            shuffle=False,
            batch_size=self.batch_size,
            classification=self.classification,
        )

        return ds_test_imbalanced, ds_test_balanced


@njit
def np_flip(arr: np.ndarray, axis=None) -> np.ndarray:
    """Flip image np arrays with a (batch, row, col, band) configuration."""
    forw = np.int64(1)
    rev = np.int64(-1)
    flip_ax0 = np.random.choice(np.array([forw, rev]))
    flip_ax1 = np.random.choice(np.array([forw, rev]))

    flips = tuple(
        (
            slice(None, None, None),  # TF batch
            slice(None, None, flip_ax0),  # image rows
            slice(None, None, flip_ax1),  # image cols
            slice(None, None, None),  # image bands
        )
    )
    return arr[flips]


def flip_batch(arr: np.ndarray) -> np.ndarray:
    return np.apply_over_axes(np_flip, arr, axes=0)


class GreenDataset(Sequence):
    def __init__(
        self,
        img=None,
        osm=None,
        eua=None,
        ulc=None,
        urb=None,
        shuffle=False,
        batch_size=256,
        classification=False,
        augment=False,
    ):
        "docstring"
        self.img = img
        self.osm = osm
        self.eua = eua
        self.ulc = ulc
        self.urb = urb
        assert self.img.shape[0] == self.osm.shape[0] == self.eua.shape[0]

        self.pred = {}
        self.indexes = None
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.classification = classification
        if augment:
            # flip if augmentation is requested ...
            logging.info("Using augmentation method np_random_flip")
            self.augment = flip_batch
        else:
            # ... bypass if not
            self.augment = lambda x: x

        self.on_epoch_end()

    def __len__(self):
        # mandatory for tf.keras.utils.Sequence
        return int(np.ceil(len(self.eua) / self.batch_size))

    def __getitem__(self, index):
        # mandatory for tf.keras.utils.Sequence
        i_start = index * self.batch_size
        i_end = min((index + 1) * self.batch_size, len(self.indexes))
        idx = self.indexes[i_start:i_end]

        img_item = self.augment(self.img[idx, :, :, :])
        osm_item = self.osm[idx, :]
        eua_item = self.eua[idx, 1]

        return {"input_img": img_item, "input_osm": osm_item}, eua_item

    @property
    def element_spec(self):
        return ({"input_img": self.img.shape[1:], "input_osm": self.osm.shape[1]}, 1)

    def on_epoch_end(self):
        # optional for tf.keras.utils.Sequence
        self.indexes = np.arange(len(self.img))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def as_numpy_iterator(self):
        for idx in self.indexes:
            yield {"input_img": self.img[idx], "input_osm": self.osm[idx]}, self.eua[idx, 1]

    def get_urb_iterator(self):
        for idx in self.indexes:
            yield self.urb[idx]

    def add_predictions(self, predictions, predictions_name):
        self.pred.update({predictions_name: predictions})

    def toggle_augment(self):
        if isinstance(self.augment, types.LambdaType):
            self.augment = flip_batch
        else:
            self.augment = lambda x: x

    def write_h5(self, filename):
        with File(filename, "x") as f:
            f.create_dataset("img", data=self.img)
            f.create_dataset("osm", data=self.osm)
            f.create_dataset("eua", data=self.eua)
            f.create_dataset("ulc", data=self.ulc)
            f.create_dataset("urb", data=self.urb)
            if len(self.pred) != 0:
                grp = f.create_group("pred")
                for name, data in self.pred.items():
                    grp.create_dataset(name, data=data)

    def _create_geometry(self, ul_x, ul_y, img_obj, tile_size):
        win_chunk = rio.windows.Window(ul_x, ul_y, *tile_size)
        geometry = box(*img_obj.window_bounds(win_chunk))
        return geometry

    def write_gpkg(self, filename, reference_raster_file, tile_size=(33, 33)):
        # open reference_raster, used to derive geometries for keys
        with rio.open(reference_raster_file) as ras:
            # create dataframe with keys and geometries
            df_keys = pd.DataFrame(
                {
                    "ulc": [f"{ul_x}_{ul_y}" for ul_x, ul_y in self.ulc],
                    "urb": self.urb,
                    "geometry": [
                        self._create_geometry(ul_x, ul_y, ras, tile_size) for ul_x, ul_y in self.ulc
                    ],
                }
            )

        if self.classification:
            df_label = pd.DataFrame({"label_class": np.argmax(self.eua, axis=1)})
        else:
            df_label = pd.DataFrame(self.eua[:, 1], columns=["public green"])

        df_all = pd.concat(
            [
                df_keys.reset_index(drop=True),
                df_label.reset_index(drop=True),
            ],
            axis=1,
        )

        if len(self.pred) != 0:
            for name, data in self.pred.items():
                if self.classification:
                    df_all = pd.concat(
                        [
                            df_all.reset_index(drop=True),
                            pd.DataFrame({f"{name}_pred_class": np.argmax(data, axis=1)}),
                        ],
                        axis=1,
                    )
                else:
                    df_all = pd.concat(
                        [
                            df_all.reset_index(drop=True),
                            pd.DataFrame(data, columns=[f"{name}_pred_public_green"]),
                        ],
                        axis=1,
                    )

        gdf_all = gpd.GeoDataFrame(df_all, crs="EPSG:3035")
        logging.debug("Writing predictions to GeoPackage.")
        gdf_all.to_file(filename, driver="GPKG")


class GreenDaskDataset(GreenDataset):
    def __init__(self, h5file, scaling_factors, batch_size=None):
        self.h5file = h5file
        self.scaling_factors = pickle.load(open(scaling_factors, "rb"))

        self.h5_file_handle = File(self.h5file, "r")

        self.img = min_max_scale(
            da.from_array(self.h5_file_handle["img"], chunks="auto"),
            self.scaling_factors["img_min"],
            self.scaling_factors["img_max"],
        )
        self.osm = min_max_scale(
            da.from_array(self.h5_file_handle["osm"], chunks="auto"),
            self.scaling_factors["osm_min"],
            self.scaling_factors["osm_max"],
        )
        self.eua = min_max_scale(
            da.from_array(self.h5_file_handle["eua"], chunks="auto")[:, 1],
            self.scaling_factors["eua_min"][:, 1],
            self.scaling_factors["eua_max"][:, 1],
        )

        self.ulc = da.from_array(self.h5_file_handle["ulc"], chunks="auto")
        self.urb = da.from_array(self.h5_file_handle["urb"], chunks="auto")

        # optimize batchsize if not provided to fit chunk size
        if batch_size:
            self.batch_size = batch_size
        else:
            self.batch_size = self.img.chunksize[0]

        self.indexes = list(range(len(self.img)))

    def __del__(self):
        self.h5_file_handle.close()

    def __len__(self):
        # mandatory for tf.keras.utils.Sequence
        return super(GreenDaskDataset, self).__len__()

    def __getitem__(self, index):
        # mandatory for tf.keras.utils.Sequence
        i_start = index * self.batch_size
        i_end = min((index + 1) * self.batch_size, len(self.img))
        idx = list(range(i_start, i_end))

        img_item = self.img[idx].compute()
        osm_item = self.osm[idx].compute()
        eua_item = self.eua[idx].compute()

        return {"input_img": img_item, "input_osm": osm_item}, eua_item

    def on_epoch_end(self):
        pass

    # @property
    # def element_spec(self):
    # return super(GreenDaskDataset, self).element_spec

    def as_numpy_iterator(self):
        # for idx in self.indexes:
        #     yield {
        #         "input_img": self.img[idx].compute(),
        #         "input_osm": self.osm[idx].compute(),
        #     }, self.eua[idx].compute()
        # THIS TAKES AGES. And it is not really required
        raise NotImplementedError


def min_max_scale(arr: np.ndarray, min: np.ndarray, max: np.ndarray):
    out_arr = np.nan_to_num(np.divide((arr - min), (max - min), dtype=np.float32))
    return out_arr


def spatialize_entire_green_precursor(
    gpc: GreenDatasetPrecursor,
    output_gpkg: str,
    reference_raster_file: str,
):
    gds = GreenDataset(
        img=gpc.img,
        osm=gpc.osm,
        eua=gpc.eua,
        urb=gpc.urb[:, 3],
        ulc=gpc.ulc,
        shuffle=False,
    )
    gds.write_gpkg(output_gpkg, reference_raster_file)


def spatialize_entire_file(filename: str, output_gpkg: str, reference_raster_file: str):
    gpc = GreenDatasetPrecursor(
        filename, balance_training=False, shuffle=False, clean_on_read=False
    )
    spatialize_entire_green_precursor(gpc, output_gpkg, reference_raster_file)
