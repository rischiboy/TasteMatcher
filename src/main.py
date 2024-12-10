import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
import yaml

from preprocessing import pipeline
from model import build_network, build_model

from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras.optimizers import Adagrad, Adam, RMSprop, SGD

from tensorflow.keras.layers import Dense, Conv2D, Input, Dropout, Flatten, Lambda
from tensorflow.keras.layers import Layer
from tensorflow.keras import backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint


"""
Sources: 
https://medium.com/@crimy/one-shot-learning-siamese-networks-and-triplet-loss-with-keras-2885ed022352
https://github.com/CrimyTheBold/tripletloss
"""

MARGIN = 0.2
PREPROCESSOR = tf.keras.applications.resnet.preprocess_input


def get_params():

    with open("config/params.yaml", "r") as file:
        params = yaml.safe_load(file)

    # Preprocessing parameters
    preprocessing_params = params["preprocessing"]
    preprocessing_params["seed"] = params["seed"]
    preprocessing_params["validation_size"] = params["validation_size"]
    preprocessing_params["preprocessor"] = PREPROCESSOR

    # Model parameters
    model_params = params["model"]

    # Training parameters
    training_params = params["training"]

    return preprocessing_params, model_params, training_params


def create_optimizer(optimizer, lr):
    if optimizer == "adam":
        return Adam(learning_rate=lr)
    elif optimizer == "adagrad":
        return Adagrad(learning_rate=lr)
    elif optimizer == "rmsprop":
        return RMSprop(learning_rate=lr)
    elif optimizer == "sgd":
        return SGD(learning_rate=lr)
    else:
        raise ValueError("Invalid optimizer")


def main():

    # Path to the data
    train_path = "data/train_triplets.txt"
    test_path = "data/test_triplets.txt"
    predictions_path = "data/results.txt"

    preprocessing_params, model_params, training_params = get_params()

    # Load the training and validation data
    train_data, val_data = pipeline(
        data_path=train_path, params=preprocessing_params, num_samples=None, train=True
    )

    # Build the predictive model
    network = build_network(model_params=model_params)
    model = build_model(
        network=network, triplet_layer_params=model_params["triplet_loss"]
    )

    # Train the model
    encoder = train_model(train_data, val_data, model, training_params)

    # Load the test data
    test_data = pipeline(data_path=test_path, params=preprocessing_params, num_samples=None, train=False)

    # Predict the results
    predictions = predict_results(test_data, encoder, predictions_path)

    return


def abs_dist(a, b):
    return np.sum(np.abs(a - b))


def squared_dist(a, b):
    return np.sum(np.square(a - b))


def validate_accuracy(anchor, positive, negative):
    assert len(anchor) == len(positive) and len(anchor) == len(negative)

    num_samples = len(anchor)
    num_correct = 0

    for a, p, n in zip(anchor, positive, negative):

        # Squared Euclidian distance
        pos_dist = squared_dist(a, p)
        neg_dist = squared_dist(a, n)

        if pos_dist < neg_dist:
            num_correct += 1

    return num_correct / num_samples


def validate_model(val_data, model):

    A = np.stack(val_data["anchor"].to_numpy())
    P = np.stack(val_data["positive"].to_numpy())
    N = np.stack(val_data["negative"].to_numpy())

    val_anchor = model.predict(A)
    val_positive = model.predict(P)
    val_negative = model.predict(N)

    val_loss = np.sum(
        np.sum(np.square(val_anchor - val_positive), axis=1)
        - np.sum(np.square(val_anchor - val_negative), axis=1)
    )
    val_accuracy = validate_accuracy(val_anchor, val_positive, val_negative)

    return val_loss, val_accuracy


def train_model(train_data, val_data, model, train_params):

    print("> Training the model")

    # Unpack the training parameters
    batch_size = train_params["batch_size"]
    epochs = train_params["epochs"]
    val_interval = train_params["val_interval"]  # epochs between validation
    optimizer = train_params["optimizer"]
    lr = train_params["learning_rate"]

    optimizer = create_optimizer(optimizer, lr)

    # Compile the model
    # model.compile(optimizer=optimizer, loss=None)
    model.compile(optimizer=optimizer, loss=None, metrics=[triple_accuracy])
    model.summary()

    encoder = model.get_layer("encoder")

    total_num_validation = epochs // val_interval

    # Callbacks
    # best_model_file = "best_model.h5"
    # early_stop = EarlyStopping(
    #     monitor="triple_accuracy", patience=5, restore_best_weights=True
    # )
    # checkpoint = ModelCheckpoint(
    #     best_model_file,
    #     monitor="triple_accuracy",
    #     save_best_only=True,
    #     save_weights_only=True,
    #     mode="max",
    # )

    # Prepare the training data in the correct format
    anchor = np.stack(train_data["anchor"].to_numpy())
    positive = np.stack(train_data["positive"].to_numpy())
    negative = np.stack(train_data["negative"].to_numpy())

    # Train loop
    for i in range(total_num_validation):

        print(f"EPOCH {i*val_interval}-{(i+1)*val_interval} out of {epochs}")

        model.fit(
            x=[anchor, positive, negative],
            batch_size=batch_size,
            epochs=val_interval,
        )

        val_loss, val_accuracy = validate_model(val_data, encoder)

        print(f"Validation Loss: {val_loss}   Validation Accuracy: {val_accuracy}")

    print("> Done")

    return encoder


def triple_accuracy(y_true, y_pred):
    return K.mean(y_pred < MARGIN)


def calc_loss(input, net):
    length = len(input)
    loss_list = []
    # print(net.input_shape)
    pred_anchor = net.predict(np.array([l[0] for l in input]))
    pred_1 = net.predict(np.array([l[1] for l in input]))
    pred_2 = net.predict(np.array([l[2] for l in input]))

    for i in range(len(pred_anchor)):
        if np.sum(np.square(pred_anchor[i] - pred_1[i])) < np.sum(
            np.square(pred_anchor[i] - pred_2[i])
        ):
            temp = 1
        else:
            temp = 0
        loss_list.append(1 - temp)

    return np.sum(loss_list) / length


def predict_results(test_data: pd.DataFrame, net: tf.keras.Model, out_file: str):
    print("> Predicting on test data")
    print(">> Number of samples: ", len(test_data))

    A = np.stack(test_data["anchor"].to_numpy())
    P = np.stack(test_data["positive"].to_numpy())
    N = np.stack(test_data["negative"].to_numpy())

    test_anchor = net.predict(A)
    test_positive = net.predict(P)
    test_negative = net.predict(N)

    predictions = []

    for a, p, n in zip(test_anchor, test_positive, test_negative):
        pos_dist = squared_dist(a, p)
        neg_dist = squared_dist(a, n)

        if pos_dist < neg_dist:
            predictions.append(1)
        else:
            predictions.append(0)

    with open(out_file, "w") as writer:
        for item in predictions:
            writer.write("%s\n" % item)

    print(">> Predictions saved to ", out_file)
    print("> Done")

    return predictions


if __name__ == "__main__":
    main()
