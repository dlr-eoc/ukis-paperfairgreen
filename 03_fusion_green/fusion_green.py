"""Provides wrappers for creating and training networks."""
import logging
from typing import List

import accuracy as acc
import custom_models as cm
import joblib
from numpy import array
from tensorflow.keras import Model
from tensorflow.keras import backend as K
from tensorflow.keras.applications import EfficientNetB3, MobileNetV2
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
    TensorBoard,
)
from tensorflow.keras.layers import (
    Activation,
    BatchNormalization,
    Dense,
    Dropout,
    GlobalAveragePooling2D,
    Input,
    concatenate,
)
from tensorflow.keras.losses import CategoricalCrossentropy, Huber
from tensorflow.keras.metrics import CategoricalAccuracy, MeanAbsoluteError, MeanSquaredError
from tensorflow.keras.optimizers import Adam


def create_ann(
    input_shape, dropout_rate=0.1, hidden_layers=0, hidden_layer_nodes: int = 10, bn_or_do="do"
):
    """Create the ANN part for the fusion network.

    Parameters
    ----------
    input_shape: tuple
        Shape of expected input data. To be inferred from input_data.shape

    Returns
    -------
    model: tf.keras.Model
        ANN model
    """
    if isinstance(hidden_layer_nodes, int):
        hidden_layer_nodes = [hidden_layer_nodes] * hidden_layers

    input = Input(shape=input_shape, name="input_osm")

    for i, n_nodes in enumerate(hidden_layer_nodes):
        if "x" not in locals():
            x = Dense(n_nodes, kernel_initializer="he_normal", name=f"ann_dense_{i}")(input)
        else:
            x = Dense(n_nodes, kernel_initializer="he_normal", name=f"ann_dense_{i}")(x)
        if bn_or_do == "do":
            x = Dropout(rate=dropout_rate, name=f"ann_dropout_{i}")(x)
        elif bn_or_do == "bn":
            x = BatchNormalization(name=f"ann_batchnorm_{i}")(x)
        x = Activation("relu", name=f"ann_relu_{i}")(x)

    # Create model
    model = Model(input, x, name="ANN")

    return model


def create_cnn(
    input_shape,
    cnn_model,
    skip_connections: bool = True,
    dropout_rate=0.1,
    cnn_dense_layers: List[int] = [10, 10, 10],
):
    """Create the CNN part for the fusion network.

    Parameters
    ----------
    input_shape : tuple[int]
        Shape of the expected input data. To be inferred from input_data.shape
    cnn_model : str
        Selection of network architecture.
    skip_connections : bool
        Should the model use skip connections. Default True

    Returns
    -------
    model: tf.keras.Model
    """
    # Create CNN
    cnn_input = Input(shape=input_shape, name="input_img")

    if cnn_model == "mobilenet":
        base_model = MobileNetV2(
            weights=None,
            input_shape=input_shape,
            include_top=False,
        )

    elif cnn_model == "efficientnetb3":
        base_model = EfficientNetB3(
            weights=None,
            input_shape=input_shape,
            include_top=False,
        )

    elif cnn_model == "greennet":
        base_model = cm.greennet_v2(
            input_shape=input_shape,
            skip_connections=skip_connections,
            dropout_rate=dropout_rate,
            cnn_dense_layers=cnn_dense_layers,
        )
    else:
        raise NotImplementedError

    cnn_output = base_model(cnn_input, training=True)
    return Model(cnn_input, cnn_output, name="CNN")


def create_fusion_net(
    dataset_train,
    mode="fusion",
    cnn_model="mobilenet",
    classification=False,
    dropout_rate=0.1,
    fusion_cnn_neurons=500,
    fusion_ann_neurons=500,
    cnn_dense_layers: List[int] = [10, 10, 10],
    ann_hidden_layers=0,
    ann_hidden_layer_nodes=10,
    ann_bn_or_do="do",
    fusion_hidden_layers=0,
    fusion_hidden_layer_nodes=10,
    greennet_skipconn=False,
    verbose=True,
):
    """Create the fusion network by combining a ANN and a CNN.

    Parameters
    ----------
    image_data: list(numpy.ndarray (n, m, d))
        List of image tiles. Used to determine the shape of input data for the CNN.
    osm_attributes: list(numpy.ndarray (n,))
        List of osm attribute data. Used to determine the shape of input data for the ANN.
    labels: list(numpy.ndarray (n,))
        List of osm attribute data. Used to determine the shape of input data for the output layer
        in the fusion network.

    Returns
    -------
    model: tf.keras.Model
        structure of the model
    """
    num_of_labels = dataset_train.element_spec[1]

    # ANN part for osm data
    if mode in ["fusion", "ann"]:
        num_of_attributes = dataset_train.element_spec[0]["input_osm"]
        ann = create_ann(
            num_of_attributes, dropout_rate, ann_hidden_layers, ann_hidden_layer_nodes, ann_bn_or_do
        )

    if mode in ["fusion", "cnn"]:
        img_shape = tuple(dataset_train.element_spec[0]["input_img"])
        cnn = create_cnn(
            img_shape,
            cnn_model,
            skip_connections=greennet_skipconn,
            dropout_rate=dropout_rate,
            cnn_dense_layers=cnn_dense_layers,
        )

    if mode == "fusion":
        # create the input to our final set of layers as the output of both MLP and CNN
        x_cnn = cnn.output
        if not cnn_model == "greennet":
            x_cnn = GlobalAveragePooling2D()(x_cnn)

        x_cnn = Dense(
            fusion_cnn_neurons,
            kernel_initializer="he_normal",
            activation="relu",
            name="FinalCNNLayer",
        )(x_cnn)

        x_ann = ann.output
        x_ann = Dense(
            fusion_ann_neurons,
            kernel_initializer="he_normal",
            activation="relu",
            name="FinalANNLayer",
        )(x_ann)

        x = concatenate([x_ann, x_cnn])

        if isinstance(fusion_hidden_layer_nodes, int):
            fusion_hidden_layer_nodes = [fusion_hidden_layer_nodes] * fusion_hidden_layers

        for i, n in enumerate(fusion_hidden_layer_nodes):
            x = Dense(n, name=f"fusion_dense_{i}")(x)
            if ann_bn_or_do == "do":
                x = Dropout(dropout_rate, name=f"fusion_dropout_{i}")(x)
            elif ann_bn_or_do == "bn":
                x = BatchNormalization(name=f"fusion_batchnorm_{i}")(x)
            x = Activation("relu", name=f"fusion_relu_{i}")(x)

        if classification:
            x = Dense(num_of_labels, activation="softmax", name="fusion_FusionClassificationLayer")(
                x
            )
        else:
            if ann_bn_or_do == "bn":
                # if batchnormalization is chosen, set one dropout in the right before the last
                # layer
                x = Dropout(dropout_rate, name=f"final_big_dropout_dropout")(x)
            x = Dense(num_of_labels, activation="linear", name="fusion_FusionRegressionLayer")(x)

        model = Model(inputs=[cnn.input, ann.input], outputs=x)

    elif mode == "ann":
        x = ann.output
        if ann_bn_or_do == "bn":
            # if batchnormalization is chosen, set one dropout in the right before the last
            # layer
            x = Dropout(dropout_rate, name=f"final_big_dropout_dropout")(x)
        if classification:
            x = Dense(num_of_labels, activation="softmax", name="ANNClassificationLayer")(x)
        else:
            x = Dense(num_of_labels, activation="linear", name="ANNRegressionLayer")(x)
        model = Model(inputs=ann.input, outputs=x)
    elif mode == "cnn":
        x = cnn.output
        if ann_bn_or_do == "bn":
            # if batchnormalization is chosen, set one dropout in the right before the last
            # layer
            x = Dropout(dropout_rate, name=f"final_big_dropout_dropout")(x)
        if not cnn_model == "greennet":
            x = GlobalAveragePooling2D()(x)
        if classification:
            x = Dense(num_of_labels, activation="softmax", name="CNNClassificationLayer")(x)
        else:
            x = Dense(num_of_labels, activation="linear", name="CNNRegressionLayer")(x)
        model = Model(inputs=cnn.input, outputs=x)
    else:
        raise NotImplementedError

    if verbose:
        model.summary()

    return model


def r2(y_true, y_pred):
    """Calculate R suqred from two Keras Tensors."""
    SS_res = K.sum(K.square(y_true - y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return 1 - SS_res / (SS_tot + K.epsilon())


def compile_network(
    model, initial_learning_rate=1e-3, classification=False, n_classes=None
) -> Model:
    """Comile the network, adding loss, evaluation metrics and optimizer.

    Parameters
    ----------
    model: tf.keras.Model
        Model to be compiled

    Returns
    -------
    model: compiled tf.keras.Model
        A compiled version of the model with added loss, evaluation metrics and optimizer
    """
    # select, loss, and metrics
    if classification:
        loss_object = CategoricalCrossentropy(name="CCE")
        metrics = [
            CategoricalAccuracy(name="OA"),
        ]
    else:
        loss_object = Huber(name="huber")

        metrics = [
            MeanAbsoluteError(name="MAE", dtype=None),
            MeanSquaredError(name="MSqE", dtype=None),
            r2,
        ]

    optimizer = Adam(learning_rate=initial_learning_rate)

    logging.info(f"weights: {len(model.weights)}")
    logging.info(f"trainable_weights: {len(model.trainable_weights)}")

    model.compile(optimizer, loss_object, metrics)

    return model


def train_network(
    compiled_model,
    model_path,
    dataset_train,
    dataset_val,
    epochs,
    warmup_epochs,
    warmup_lr,
    reduce_on_plateau_patience,
    early_stopping_patience,
    log_dir,
):
    """Train the model using tf.datasets.

    Parameters
    ----------
    compiled_model: tf.Model
        Model compiled by compile_network()

    model_path: str
        file path to which the model will be saved.

    dataset_train: tf.Dataset
        A tf.Dataset containing training data and labels

    dataset_val: tf.Dataset
        A tf.Dataset containing validation data and labels

    epochs: int
        Number of epochs to train for


    Returns
    -------
    history: tf.Model.history
         Model history of the trained model
    """
    callbacks = [
        # saves new weights only when the model is better than before
        ModelCheckpoint(
            model_path,
            monitor="val_loss",
            save_weights_only=True,
            save_best_only=True,
            mode="min",
            verbose=1,
        ),
        # Start with high learning rate and reduce when val-loss not better -> yields higher gain in
        # the beginning
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.2,
            patience=reduce_on_plateau_patience,
            min_lr=0.0000001,
            verbose=1,
        ),
        # method to avoid overfitting. when val_loss does not get better after 5 epochs, saves time
        # when training for higher numbers of epochs
        EarlyStopping(
            monitor="val_loss",
            min_delta=0,
            patience=early_stopping_patience,
            verbose=1,
            mode="auto",
            baseline=None,
            restore_best_weights=True,
        ),
    ]

    # add tensorboard callback if required
    if log_dir:
        callbacks += [
            TensorBoard(
                log_dir=log_dir,
                histogram_freq=1,
            )
        ]

    if warmup_epochs > 0:
        # change learning rate
        prior_lr = float(compiled_model.optimizer.lr)
        K.set_value(compiled_model.optimizer.lr, warmup_lr)
        wu_cb = [
            ModelCheckpoint(model_path, save_weights_only=True, save_best_only=False, verbose=1)
        ]
        if log_dir:
            wu_cb += [
                TensorBoard(
                    log_dir=log_dir,
                    histogram_freq=1,
                )
            ]

        compiled_model.fit(
            dataset_train,
            epochs=warmup_epochs,
            callbacks=wu_cb,
        )
        # revert learning rate to compiled state
        K.set_value(compiled_model.optimizer.lr, prior_lr)

    history = compiled_model.fit(
        dataset_train,
        epochs=epochs,
        callbacks=callbacks,
        validation_data=dataset_val,
        initial_epoch=warmup_epochs,
        use_multiprocessing=True,
    )

    logging.info(f"Model trained with parameters: {history.params}")
    logging.info(f"Model Trained for {len(history.history['loss'])} epochs.")

    return history


def infer_network(compiled_model, model_path, dataset, predictions_name):
    """Infer the compiled network on new data.

    Parameters
    ----------
    compiled_model: tf.Model
        model compiled by compile_network() and trained by train_network()

    dataset: tf.Dataset
        Dataset for which the inference should take place.

    predictions_name: str
        name of prediction set serving as identifier in the dataset's prediction dict

    Returns
    -------
    None
    """
    compiled_model.load_weights(model_path)
    pred = compiled_model.predict(dataset, use_multiprocessing=True)
    dataset.add_predictions(pred, predictions_name)

    return None


def train_and_evaluate_sklearn(
    learner, dataset_train, datasets_test_dict, experiment_title, experiment_path, classification
):
    """Train and evaluate a sklearn regressor.

    Parameters
    ----------
    dataset_train: tf.Dataset
        Dataset for which the inference should take place.

    datasets_test_dict: dict[tf.Dataset, tf.Dataset]
        Dictionary of different datasets the RandomForestRegressor should be evaluated against.
        Intended to supply, e.g., balanced and imbalanced datasets

    learner:
        sklearn model providing both a learner.fit and learner.predict method.

    predictions_name: str
        name of prediction set serving as identifier in the dataset's prediction dict

    Returns
    -------
    None
    """
    osm_train = array(list(x[0]["input_osm"] for x in dataset_train.as_numpy_iterator()))
    labels_train = array(list(x[1] for x in dataset_train.as_numpy_iterator()))

    learner.fit(osm_train, labels_train)

    joblib.dump(learner, f"{experiment_path}/model_{experiment_title}.pkl")

    for bal_imbal, dataset in datasets_test_dict.items():
        # extract osm information from dataset
        osm_test = array(list(x[0]["input_osm"] for x in dataset.as_numpy_iterator()))

        # extract labels from dataset
        y_labels = array(list(x[1] for x in dataset.as_numpy_iterator()))

        # extract urban_proximity from dataset
        y_urb_prox = array([i for i in dataset.get_urb_iterator()])

        # predict the model
        y_pred = learner.predict(osm_test)

        if classification:
            # overall accuracies
            acc_df = acc.df_accuracies(y_labels, y_pred)
            acc_df[0].to_csv(f"{experiment_path}/conf_mat_{experiment_title}_{bal_imbal}.csv")
            acc_df[1].to_csv(f"{experiment_path}/class_acc_{experiment_title}_{bal_imbal}.csv")
            acc_df[1].to_csv(f"{experiment_path}/class_acc_{experiment_title}_{bal_imbal}.csv")

            # assess accuracies
            acc_df_urb_gradient = acc.df_accuracies_urban_gradient(
                y_labels, y_pred, y_urb_prox, classification
            )

            acc_df_urb_gradient[0].to_csv(
                f"{experiment_path}/conf_mat_urb_gradient_{experiment_title}_{bal_imbal}.csv"
            )
            acc_df_urb_gradient[1].to_csv(
                f"{experiment_path}/class_acc_urb_gradient_{experiment_title}_{bal_imbal}.csv"
            )

            logging.info(f"\nConfusion Matrix {experiment_title} {bal_imbal}:\n{acc_df[0]}\n")
            logging.info(f"\nAccuracy metrics {experiment_title} {bal_imbal}:\n{acc_df[1]}\n")

            # plot accuracies
            acc.plot_classification(y_labels, y_pred, experiment_path, experiment_title, bal_imbal)
            acc.plot_class_based_accuracies(acc_df[1], experiment_path, experiment_title, bal_imbal)
            acc.plot_classification_urban_gradient(
                acc_df_urb_gradient[1], experiment_path, experiment_title, bal_imbal
            )

            # add predictions to dataset
            dataset.add_predictions(y_pred, experiment_title)
        else:
            acc_df = acc.df_accuracies_green_gradient(y_labels, y_pred, classification)
            # write the accuracies to file
            acc_df.to_csv(f"{experiment_path}/acc_{experiment_title}_{bal_imbal}.csv")

            # log aggregate information
            acc.log_aggregate_accuracies(acc_df, experiment_title, bal_imbal)

            # plot accuracies
            acc.plot_regression(acc_df, experiment_path, experiment_title, bal_imbal)

            # add predictions to dataset
            dataset.add_predictions(y_pred, experiment_title)
