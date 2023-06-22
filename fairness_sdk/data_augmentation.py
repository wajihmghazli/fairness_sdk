import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split


def RAFDB_train_augmentation(train_images: np.ndarray, train_labels: np.ndarray) -> ImageDataGenerator:
    """
    Apply data augmentation to the training images using the RAF-DB dataset.

    Args:
        train_images (np.ndarray): The training images.
        train_labels (np.ndarray): The corresponding training labels.

    Returns:
        ImageDataGenerator: An instance of the ImageDataGenerator class configured with the desired data augmentation parameters.
    """
    # Split the data into training and validation sets
    train_images, val_images, train_labels, val_labels = train_test_split(train_images, train_labels, test_size=0.2, random_state=42)

    # Create an instance of the ImageDataGenerator class with the desired data augmentation parameters
    datagen = ImageDataGenerator(
        rotation_range=20, # Rotate images randomly by 20 degrees
        width_shift_range=0.1, # Shift images horizontally by up to 10% of the width
        height_shift_range=0.1, # Shift images vertically by up to 10% of the height
        shear_range=0.2, # Apply shearing transformation with a maximum shear of 20%
        zoom_range=0.2, # Randomly zoom in and out by 20%
        horizontal_flip=True, # Flip images horizontally
        fill_mode='nearest' # Fill any empty pixels with the nearest value
    )

    # Fit the ImageDataGenerator to the training data
    datagen.fit(train_images)
    return datagen
