import os
from typing import Callable, Dict, List, Optional, Tuple
import tensorflow as tf
from tqdm import tqdm
from PIL import Image
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import yaml


"""
Resizes and saves the food images to a new directory
"""


def resizeImages(image_dir: str, out_dir: str, height: int, width: int):

    print("Resize images")

    if os.path.isdir(out_dir):
        print("Images already resized")
        return

    # Create the output directory
    os.makedirs(out_dir)
    included_extensions = ["jpg", "jpeg", "png", "gif"]
    images_list = [
        fn
        for fn in os.listdir(image_dir)
        if any(fn.endswith(ext) for ext in included_extensions)
    ]
    images_list = sorted(images_list)

    for img in tqdm(images_list):
        temp = Image.open(os.path.join(image_dir, img))
        temp = temp.resize((height, width))
        temp.save(os.path.join(out_dir, img))

    return


def load_encodings(triplets: List, encondings: List):
    images = []

    for i in range(len(triplets)):
        temp = []
        for j in range(3):
            id = triplets[i][j]
            img = encondings[id]
            temp.append(img)
        images.append(temp)
    return images


def encode_images(image_dir: str, encoding_file: str, encoder: tf.keras.Model):
    extensions = ["jpg", "jpeg", "png", "gif"]
    image_files = [
        fn
        for fn in os.listdir(image_dir)
        if any(fn.endswith(ext) for ext in extensions)
    ]
    # train_files = sorted(train_files)

    images = []
    for file in tqdm(image_files):
        img = Image.open(os.path.join(image_dir, file))
        # Normalize the image
        img = np.array(img) / 255.0
        img = img.astype(np.float32)
        images.append(img)

    print("> Encoding images")
    result = encoder.predict(np.array(images))

    with open(encoding_file, "w") as writer:
        for item in result:
            item = item.tolist()
            writer.write("%s\n" % item)

    result = np.array(result)
    return result


def load_images(triplets: pd.DataFrame, image_dir: str) -> pd.DataFrame:
    """
    Load the images from the directory and return them as a list.

    Args:
        triplets: Dataframe with the triplets of image names (anchor, positive, negative)
        image_dir: Directory where the images are stored

    Returns:
        images: Dataframe of triplets of images (anchor, positive, negative)
    """

    # Dictionary to store loaded images
    image_cache = {}

    # Helper function to load an image if not already loaded
    def get_image(image_name):
        if image_name not in image_cache:
            # Load the image and store it in the cache
            image_path = f"{image_dir}/{image_name}.jpg"
            image_cache[image_name] = Image.open(image_path)
        return image_cache[image_name]

    # Replace image names with actual image data
    images = triplets.map(get_image)

    return images


def preprocess_images(
    images: pd.DataFrame,
    resize_shape: Tuple[int, int],
    preprocessor: Optional[Callable] = None,
) -> List:
    """
    Prepares the images to be fed into the encoder network.

    Args:
        triplets: List of triplets of images (anchor, positive, negative)
        preprocessor: Preprocessing function to apply to the images according to the encoder network (ResNet, Inception, etc.)

    Returns:
        List of preprocessed images
    """

    # Resize the images
    images = images.map(lambda x: x.resize(resize_shape))

    if preprocessor is None:
        normalized_images = images.map(lambda x: x / 255.0)
        return normalized_images

    def preprocess_column(column):
        # Stack all images in the column into a batch for preprocessing
        stacked_images = np.stack(column.values)
        preprocessed_images = preprocessor(stacked_images, data_format=None)
        preprocessed_images /= 255.0  # Normalize the images
        return pd.Series(list(preprocessed_images))

    # Apply the preprocessing to all columns of the DataFrame (anchor, positive, negative)
    preprocessed_images = images.apply(preprocess_column, axis=0)
    return preprocessed_images


def training_validation_split(
    data: pd.DataFrame,
    validation_size: float = 0.01,
    seed: int = 42,
    filter: bool = False,
    save: bool = False,
):

    # Split the data into training and validation
    train, val = train_test_split(data, test_size=validation_size, random_state=seed)

    if filter:
        # Get the unique images in the training and validation set
        unique_training_imgs = pd.unique(train.values.flatten()).tolist()
        unique_validation_imgs = pd.unique(val.values.flatten()).tolist()

        print(f">> # unique images in the training set: {len(unique_training_imgs)}")
        print(
            f">> # unique images in the validation set: {len(unique_validation_imgs)}"
        )

        # Get the images that are only in the training set
        only_training_imgs = list(
            set(unique_training_imgs) - set(unique_validation_imgs)
        )
        print(f">> # unique images only in the training set: {len(only_training_imgs)}")

        # Remove images from the training set that are in the validation set
        train = train[
            (train["anchor"].isin(only_training_imgs))
            & (train["positive"].isin(only_training_imgs))
            & (train["negative"].isin(only_training_imgs))
        ]

    if save:
        data_dir = "data"
        train.to_csv(f"{data_dir}/train.csv", index=False)
        val.to_csv(f"{data_dir}/val.csv", index=False)

    return train, val


def pipeline(
    data_path: str, params: Dict, num_samples: Optional[int] = None, train: bool = True
):
    print("> Preprocessing")

    seed = params["seed"]
    height = params["height"]
    width = params["width"]
    preprocessor = params["preprocessor"]

    # Load the data
    data = pd.read_csv(
        data_path,
        sep=" ",
        dtype=str,
        header=None,
        names=["anchor", "positive", "negative"],
    )

    if num_samples is not None:
        data = data.head(num_samples)  # Limit the number of samples for testing

    # Split the data into training and validation
    if train:
        validation_size = params["validation_size"]

        train_data, val_data = training_validation_split(
            data=data,
            validation_size=validation_size,
            seed=seed,
            filter=True,
            save=True,
        )

        train_images = load_images(train_data, "food")
        val_images = load_images(val_data, "food")

        # Preprocess the images
        train_images = preprocess_images(
            images=train_images, resize_shape=(width, height), preprocessor=preprocessor
        )
        val_images = preprocess_images(
            images=val_images, resize_shape=(width, height), preprocessor=preprocessor
        )

        res = (train_images, val_images)

    else:
        test_images = load_images(data, "food")
        test_images = preprocess_images(
            images=test_images, resize_shape=(width, height), preprocessor=preprocessor
        )

        res = test_images

    print("> Done")

    return res


if __name__ == "__main__":

    with open("config/params.yaml", "r") as file:
        params = yaml.safe_load(file)

    SEED = params["seed"]
    VALIDATION_SIZE = params["validation_size"]
    HEIGHT = params["preprocessing"]["height"]
    WIDTH = params["preprocessing"]["width"]
    PREPROCESSOR = tf.keras.applications.resnet.preprocess_input

    ###### Resize images ######

    # image_dir = "food"
    # out_dir = "food_resized"

    # resizeImages(image_dir, out_dir, HEIGHT, WIDTH)
