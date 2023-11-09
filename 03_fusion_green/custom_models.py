from typing import List, Tuple

from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Activation,
    BatchNormalization,
    Concatenate,
    Conv2D,
    Dense,
    Dropout,
    Flatten,
    GlobalAveragePooling2D,
    Input,
    MaxPooling2D,
)


def greennet_v2(
    input_shape: Tuple[int, int, int] = (33, 33, 5),
    dropout_rate: float = 0.1,
    skip_connections: bool = False,
    cnn_dense_layers: List[int] = [10, 10, 10],
):
    """Implement a custom VGG like simple CNN for use in green space fusion network study.

    Parameters
    ----------
    input_shape: tuple
        Input tensor shape of multilayer image data
    dropout_rate: float
        Dropout rate used within the dense layers of the network


    Returns
    -------
    tf.keras.Model
    """
    if len(cnn_dense_layers) != 3:
        raise ValueError("cnn_hidden_layers must be list of length 3")

    input_ = Input(shape=input_shape)
    con_1_1 = Conv2D(16, (9, 9), kernel_initializer="he_normal", padding="same", name="conv_1_1")
    bn_1_1 = BatchNormalization(name="bn_1_1")
    act_1_1 = Activation("relu", name="cnn_relu_1_1")
    con_1_2 = Conv2D(16, (9, 9), kernel_initializer="he_normal", padding="same", name="conv_1_2")
    bn_1_2 = BatchNormalization(name="bn_1_2")
    act_1_2 = Activation("relu", name="cnn_relu_1_2")
    maxpool_1 = MaxPooling2D((2, 2), name="maxpool_1")
    globavgpool2d_1 = GlobalAveragePooling2D(name="globavgpool2d_1")

    con_2_1 = Conv2D(32, (5, 5), kernel_initializer="he_normal", padding="same", name="conv_2_1")
    bn_2_1 = BatchNormalization(name="bn_2_1")
    act_2_1 = Activation("relu", name="cnn_relu_2_1")
    con_2_2 = Conv2D(32, (5, 5), kernel_initializer="he_normal", padding="same", name="conv_2_2")
    bn_2_2 = BatchNormalization(name="bn_2_2")
    act_2_2 = Activation("relu", name="cnn_relu_2_2")
    maxpool_2 = MaxPooling2D((2, 2), name="maxpool_2")
    globavgpool2d_2 = GlobalAveragePooling2D(name="globavgpool2d_2")

    con_3_1 = Conv2D(64, (3, 3), kernel_initializer="he_normal", padding="same", name="conv_3_1")
    bn_3_1 = BatchNormalization(name="bn_3_1")
    act_3_1 = Activation("relu", name="cnn_relu_3_1")
    con_3_2 = Conv2D(64, (3, 3), kernel_initializer="he_normal", padding="same", name="conv_3_2")
    bn_3_2 = BatchNormalization(name="bn_3_2")
    act_3_2 = Activation("relu", name="cnn_relu_3_2")
    maxpool_3 = MaxPooling2D((2, 2), name="maxpool_3")
    globavgpool2d_3 = GlobalAveragePooling2D(name="globavgpool2d_3")

    dense_1 = Dense(cnn_dense_layers[0], kernel_initializer="he_normal", name="greennet_dense_1")
    dropout_1 = Dropout(rate=dropout_rate, name="cnn_bn1")
    activation_1 = Activation("relu", name="cnn_relu_act1")
    dense_2 = Dense(cnn_dense_layers[1], kernel_initializer="he_normal", name="greennet_dense_2")
    dropout_2 = Dropout(rate=dropout_rate, name="cnn_bn2")
    activation_2 = Activation("relu", name="cnn_relu_act2")
    dense_3 = Dense(cnn_dense_layers[2], kernel_initializer="he_normal", name="greennet_dense_3")
    dropout_3 = Dropout(rate=dropout_rate, name="cnn_bn3")
    activation_3 = Activation("relu", name="cnn_relu_act3")
    concat = Concatenate(name="final_greennet_concat")

    if skip_connections:
        x_1_1 = con_1_1(input_)
        x_1_2 = bn_1_1(x_1_1)
        x_1_3 = act_1_1(x_1_2)
        x_1_4 = con_1_2(x_1_3)
        x_1_5 = bn_1_2(x_1_4)
        x_1_6 = act_1_2(x_1_5)
        x_1_7 = maxpool_1(x_1_6)

        x_2_1 = con_2_1(x_1_7)
        x_2_2 = bn_2_1(x_2_1)
        x_2_3 = act_2_1(x_2_2)
        x_2_4 = con_2_2(x_2_3)
        x_2_5 = bn_2_2(x_2_4)
        x_2_6 = act_2_2(x_2_5)
        x_2_7 = maxpool_2(x_2_6)

        x_3_1 = con_3_1(x_2_7)
        x_3_2 = bn_3_1(x_3_1)
        x_3_3 = act_3_1(x_3_2)
        x_3_4 = con_3_2(x_3_3)
        x_3_5 = bn_3_2(x_3_4)
        x_3_6 = act_3_2(x_3_5)
        x_3_7 = maxpool_3(x_3_6)

        x_gap_1 = globavgpool2d_1(x_1_7)
        x_gap_2 = globavgpool2d_2(x_2_7)
        x_gap_3 = globavgpool2d_3(x_3_7)

        x_4_1_1 = dense_1(x_gap_1)
        x_4_1_2 = dropout_1(x_4_1_1)
        x_4_1_3 = activation_1(x_4_1_2)

        x_4_2_1 = dense_2(x_gap_2)
        x_4_2_2 = dropout_2(x_4_2_1)
        x_4_2_3 = activation_2(x_4_2_2)

        x_4_3_1 = dense_3(x_gap_3)
        x_4_3_2 = dropout_3(x_4_3_1)
        x_4_3_3 = activation_3(x_4_3_2)

        output = concat([x_4_1_3, x_4_2_3, x_4_3_3])

        model = Model(inputs=[input_], outputs=[output], name="greennet_v2_skipcon")
    else:
        x = con_1_1(input_)
        x = bn_1_1(x)
        x = act_1_1(x)
        x = con_1_2(x)
        x = bn_1_2(x)
        x = act_1_2(x)
        x = maxpool_1(x)

        x = con_2_1(x)
        x = bn_2_1(x)
        x = act_2_1(x)
        x = con_2_2(x)
        x = bn_2_2(x)
        x = act_2_2(x)
        x = maxpool_2(x)

        x = con_3_1(x)
        x = bn_3_1(x)
        x = act_3_1(x)
        x = con_3_2(x)
        x = bn_3_2(x)
        x = act_3_2(x)
        x = maxpool_3(x)
        x = globavgpool2d_1(x)

        x = dense_1(x)
        x = dropout_1(x)
        output = activation_1(x)

        model = Model(inputs=[input_], outputs=[output], name="greennet_v2")

    return model
