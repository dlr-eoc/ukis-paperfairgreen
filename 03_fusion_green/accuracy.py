"""Accuracy Assessment Helpers in Fusion Green."""
import logging

import matplotlib.pyplot as plt
import seaborn as sns
from numpy import arange, argmax, array
from numpy import max as np_max
from numpy import min as np_min
from numpy import sum as np_sum
from pandas import DataFrame, concat
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)


def df_accuracies(label, prediction):
    """Overall accuracy metrics for classification experiments.

    Parameters
    ----------
    label : array
        Array of true values, i.e. reference.
    prediction : array
        Array of predicted values.

    Returns
    -------
    df_cm : DataFrame
        Confusion Matrix of predicted classes
    """
    lab = argmax(label, axis=1)
    pred = argmax(prediction, axis=1)

    # confusion matrix
    df_cm = DataFrame(confusion_matrix(lab, pred, labels=arange(label.shape[1])))

    df_class = concat(
        [
            DataFrame({"OA": [accuracy_score(lab, pred)] + [None] * (label.shape[1] - 1)}),
            DataFrame(
                {
                    "F1": f1_score(lab, pred, average=None, zero_division=0),
                    "Precision": precision_score(lab, pred, average=None, zero_division=0),
                    "Recall": recall_score(lab, pred, average=None, zero_division=0),
                }
            ),
        ],
        axis=1,
    )
    return df_cm, df_class


def df_r2(label, prediction, index):
    """Calculate R² for regression tasks."""
    return DataFrame(
        {
            # standard metrics
            "r2": r2_score(label, prediction),
            "mae": mean_absolute_error(label, prediction),
            "rmse": mean_squared_error(label, prediction, squared=False),
            # some stats
            "n_obs": label.shape[0],
            "min_label": np_min(label),
            "max_label": np_max(label),
            "min_pred": np_min(prediction),
            "max_pred": np_max(prediction),
            "g_a_ratio": sum(prediction[:]) / sum(label[:]),
        },
        index=[index],
    )


def df_accuracies_urban_gradient(label, prediction, urban_proximity, classification: bool):
    """Calculate the classification accuracies along the gradient of urban proximity."""
    cutoffs = range(int(np_max(urban_proximity)), int(np_min(urban_proximity)), -100)
    logging.debug(f"Using the following urban proximity cutoffs: {list(cutoffs)}")

    if classification:

        def calc_grad_acc(cutoff):
            urb_filter = urban_proximity <= cutoff

            # filter to current cutoff
            label_filt = label[urb_filter]
            pred_filt = prediction[urb_filter]

            df_cm, df_class = df_accuracies(label_filt, pred_filt)
            df_cm["urb_cutoff"] = cutoff
            df_class["urb_cutoff"] = cutoff

            return df_cm, df_class

        tmp_list = [calc_grad_acc(c) for c in cutoffs]
        df_cm_gradient = concat([i[0] for i in tmp_list])
        df_class_gradient = concat([i[1] for i in tmp_list])

        return df_cm_gradient, df_class_gradient

    else:

        def calc_grad_r2(cutoff):
            urb_filter = urban_proximity <= cutoff

            # filter to current cutoff
            label_filt = label[urb_filter]
            pred_filt = prediction[urb_filter]

            df = df_r2(label_filt, pred_filt, f"{cutoff} urb")

            df["urb_cutoff"] = cutoff

            return df

        tmp_list = [calc_grad_r2(c) for c in cutoffs]
        df_r2_gradient = concat(tmp_list)

        return df_r2_gradient


def df_accuracies_green_gradient(label, prediction, classification: bool):
    """Claculate accuracies along the gradient of urban green cover."""
    # define cutoffs used gradually narrow the dataset to certain levels of green coverage
    # these subsets will be defined using the green coverage values of BOTH label AND prediction
    cutoffs = [0, 0.25, 0.5, 0.75, 0.85, 0.9, 0.95]

    if classification:
        raise NotImplementedError
    else:
        #
        l_df_lab_gt_cutoff = [df_r2(label, prediction, "overall")]
        l_df_pred_gt_cutoff = []

        for c in cutoffs:
            # check which of the green LABELs are greater than c
            idx_lab_gt_c = label > c
            # are there any indexes left fitting this constraint
            if np_sum(idx_lab_gt_c) > 0:
                # append the values for the current
                l_df_lab_gt_cutoff.append(
                    df_r2(label[idx_lab_gt_c], prediction[idx_lab_gt_c], f"{c} lab")
                )

            # check which of the predicted green values are greater than c
            idx_pred_gt_c = prediction > c
            # are there any indexes left fitting this constraint
            if np_sum(idx_pred_gt_c) > 0:
                l_df_pred_gt_cutoff.append(
                    df_r2(label[idx_pred_gt_c], prediction[idx_pred_gt_c], f"{c} pred")
                )

        # if, for some reason, these lists were empty, assign none
        if len(l_df_lab_gt_cutoff) > 0:
            df_lab_gt = concat(l_df_lab_gt_cutoff)
        else:
            df_lab_gt = None
        if len(l_df_pred_gt_cutoff) > 0:
            df_pred_gt = concat(l_df_pred_gt_cutoff)
        else:
            df_pred_gt = None

        df_gt = concat([df_lab_gt, df_pred_gt])
        return df_gt


def log_aggregate_accuracies(acc_df, experiment_title, bal_imbal):
    """Log accuracy metrics."""
    for i in acc_df.index:
        logging.info(f"{experiment_title} {bal_imbal}: {i:9} R² {acc_df['r2'][i]:2.4f}")


def plot_regression(acc_df, experiment_path, experiment_title, bal_imbal):
    """Plot regression accuracies."""
    plot_file_path = f"{experiment_path}/plt_{experiment_title}_{bal_imbal}.png"
    logging.debug(f"Saving plot {plot_file_path}")

    # ~~~~~~ R² Barplot ~~~~~~~~~~~~~#
    # stolen from https://matplotlib.org/stable/gallery/lines_bars_and_markers/barchart.html
    # set up data
    labels = acc_df.index
    r2_gre = acc_df["r2"]
    x = arange(len(labels))
    width = 0.35

    # assemble plot
    fig, ax = plt.subplots()

    rects = ax.bar(x + (width / 2), r2_gre, width, label="R²", color="green")

    ax.legend()
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45)
    ax.set_ylim(-1, 1)
    ax.bar_label(rects, padding=3, rotation=45, fmt="%.3f")

    fig.tight_layout()

    plt.savefig(plot_file_path)
    plt.close()


def plot_classification(label, pred, experiment_path, experiment_title, bal_imbal):
    """Plot classification accuracies."""
    plot_file_path = f"{experiment_path}/plt_confmat_{experiment_title}_{bal_imbal}.png"
    logging.debug(f"Saving plot {plot_file_path}")
    ConfusionMatrixDisplay.from_predictions(argmax(label, axis=1), argmax(pred, axis=1))
    plt.savefig(plot_file_path)
    plt.close()


def plot_class_based_accuracies(acc_df, experiment_path, experiment_title, bal_imbal):
    """Plot class based accuracies for classification tasks."""
    plot_file_path = f"{experiment_path}/plt_prec-recall_{experiment_title}_{bal_imbal}.png"
    logging.debug(f"Saving plot {plot_file_path}")

    labels = acc_df.index
    precision = acc_df["Precision"]
    recall = acc_df["Recall"]
    x = arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - (width / 2), precision, width, label="Precision", color="orange")
    rects2 = ax.bar(x + (width / 2), recall, width, label="Recall", color="red")

    ax.legend()
    ax.set_xticks(x)
    ax.set_ylim(0, 1)
    ax.bar_label(rects1, padding=3, rotation=45, fmt="%.3f")
    ax.bar_label(rects2, padding=3, rotation=45, fmt="%.3f")

    fig.tight_layout()
    plt.savefig(plot_file_path)
    plt.close()


def plot_classification_urban_gradient(acc_df, experiment_path, experiment_title, bal_imbal):
    """Plot classification along the urban gradient."""
    plot_file_path = (
        f"{experiment_path}/plt_prec-recall_urb_gradient_{experiment_title}_{bal_imbal}.png"
    )
    logging.debug(f"Saving plot {plot_file_path}")

    acc_df["class"] = acc_df.index
    acc_df["urb_cutoff"] = acc_df["urb_cutoff"].astype(str)

    acc_df_long = acc_df.loc[:, ["Precision", "Recall", "urb_cutoff", "class"]].melt(
        id_vars=["class", "urb_cutoff"],
    )
    sns.lineplot(
        data=acc_df_long,
        x="urb_cutoff",
        y="value",
        hue="class",
        style="variable",
        palette=sns.color_palette("viridis", as_cmap=True),
    )
    plt.savefig(plot_file_path)
    plt.close()


def assess_model_accuracy(
    compiled_model, model_path, experiment_title, dataset_test_dict, experiment_path, classification
):
    """Perform end2end accuracy assessment.

    Parameters
    ----------
    compiled_model: tf.Model
        model compiled by compile_network() and trained by train_network()
    model_path: str
        File path from wich to load the model
    experiment_title: str
        Description of the experiments
    dataset_test_dict: dict of tf.Dataset
        Dictionary of test datasets to assess model accuracy with
    experiment_path: str
        Path to the experiment. Targetfolder for saving files
    classification: bool
        Is the experiment a classification or regression task? For classification, confusion matrix
        and class based accuracies are calcualted, regressions are evaluated using R²

    Returns
    -------
    None

    """
    # load model
    compiled_model.load_weights(model_path)

    for bal_imbal, dataset in dataset_test_dict.items():
        # predict the model
        y_pred = compiled_model.predict(dataset).reshape(-1)

        # extract labels from dataset
        y_labels = array(list(x[1] for x in dataset.as_numpy_iterator()))

        # extract urban_proximity from dataset
        y_urb_prox = array([i for i in dataset.get_urb_iterator()])

        # predictions batched data might be shorter than original data (due to batched dataset)
        if len(y_pred) != len(y_labels):
            # if so, trim to be able to calculate r2_score etc.
            y_labels = y_labels[: len(y_pred)]

        # calculate accuracies
        if classification:
            acc_cm, acc_class = df_accuracies(y_labels, y_pred)
            acc_cm.to_csv(f"{experiment_path}/conf_mat_{experiment_title}_{bal_imbal}.csv")
            acc_class.to_csv(f"{experiment_path}/class_acc_{experiment_title}_{bal_imbal}.csv")

            acc_df_urb_gradient = df_accuracies_urban_gradient(
                y_labels, y_pred, y_urb_prox, classification
            )
            acc_df_urb_gradient[0].to_csv(
                f"{experiment_path}/conf_mat_urb_gradient_{experiment_title}_{bal_imbal}.csv"
            )
            acc_df_urb_gradient[1].to_csv(
                f"{experiment_path}/class_acc_urb_gradient_{experiment_title}_{bal_imbal}.csv"
            )

            logging.info(f"\nConfusion Matrix {experiment_title} {bal_imbal}:\n{acc_cm}\n")
            logging.info(f"\nAccuracy metrics {experiment_title} {bal_imbal}:\n{acc_class}\n")

            # plot accuracies
            plot_classification(y_labels, y_pred, experiment_path, experiment_title, bal_imbal)
            plot_class_based_accuracies(acc_class, experiment_path, experiment_title, bal_imbal)
            plot_classification_urban_gradient(
                acc_df_urb_gradient[1], experiment_path, experiment_title, bal_imbal
            )

        else:
            acc_r2 = df_accuracies_green_gradient(y_labels, y_pred, classification)
            # write accuracies to file
            acc_r2.to_csv(f"{experiment_path}/acc_{experiment_title}_{bal_imbal}.csv")
            # log aggregate information
            log_aggregate_accuracies(acc_r2, experiment_title, bal_imbal)
            # plot accuracies
            plot_regression(acc_r2, experiment_path, experiment_title, bal_imbal)
