from typing import Dict, Optional
import yaml

import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Dropout, Lambda
from tensorflow.keras.applications import ResNet101
from tensorflow.keras.applications import ResNet101, ResNet152
from tensorflow.keras.applications.inception_v3 import InceptionV3


class TripletLossLayer(Layer):
    def __init__(self, alpha, **kwargs):
        self.alpha = alpha
        super(TripletLossLayer, self).__init__(**kwargs)

    def triplet_loss(self, inputs):
        anchor, positive, negative = inputs
        p_dist = K.sum(K.square(anchor - positive), axis=-1)
        n_dist = K.sum(K.square(anchor - negative), axis=-1)
        return K.sum(K.maximum(p_dist - n_dist + self.alpha, 0), axis=0)

    def call(self, inputs):
        loss = self.triplet_loss(inputs)
        self.add_loss(loss)
        return loss


def build_network(model_params: Dict) -> tf.keras.Model:

    # Build the encoder
    encoder_cfg = model_params["encoder"]
    encoder_model = eval(encoder_cfg["name"])
    trainable = encoder_cfg["trainable"]

    encoder_cfg.pop("name")
    encoder_cfg.pop("trainable")

    encoder = encoder_model(**encoder_cfg)

    # Replace the classification layer with a flatten layer
    output = encoder.layers[-1].output
    output = tf.keras.layers.Flatten()(output)

    encoder = Model(encoder.input, outputs=output, name=encoder.name)

    if not trainable:
        for layer in encoder.layers:
            layer.trainable = False

    # encoder.summary()

    # Add Fully Connected Layers to the output of the encoder
    network = Sequential(name="encoder")
    network.add(encoder)

    fc_layer_cfg = model_params["fc_layers"]

    for units in fc_layer_cfg["num_units"]:
        network.add(Dense(units, activation="relu"))
        network.add(Dropout(rate=fc_layer_cfg["dropout"]))

    # Add the final layer with the embeddingsize
    output_shape = fc_layer_cfg["output_shape"]
    network.add(Dense(output_shape, activation=None))

    # Force the encoding to live on the d-dimentional hypershpere
    network.add(Lambda(lambda x: K.l2_normalize(x, axis=-1)))

    return network


def build_model(network: tf.keras.Model, triplet_layer_params: Dict) -> tf.keras.Model:

    # Define the input layers of the siamese network
    input_shape = network.layers[0].input_shape[1:]

    anchor_input = tf.keras.layers.Input(input_shape, name="anchor")
    pos_input = tf.keras.layers.Input(input_shape, name="positive")
    neg_input = tf.keras.layers.Input(input_shape, name="negative")

    anchor_embeddings = network(anchor_input)
    pos_embeddings = network(pos_input)
    neg_embeddings = network(neg_input)

    # This layer computes the triplet loss between anchor, positive and negative embedding to determine the similarity
    triplet_loss_layer = TripletLossLayer(**triplet_layer_params)(
        [anchor_embeddings, pos_embeddings, neg_embeddings]
    )

    # Set up the end-to-end model
    model = Model(
        inputs=[anchor_input, pos_input, neg_input],
        outputs=triplet_loss_layer,
        name="Siamese",
    )

    return model


if __name__ == "__main__":

    with open("config/params.yaml", "r") as file:
        params = yaml.safe_load(file)

    model_params = params["model"]
    triplet_layer_params = params["model"]["triplet_loss"]

    network = build_network(model_params=model_params)
    network.summary()

    model = build_model(network=network, triplet_layer_params=triplet_layer_params)
    model.summary()
