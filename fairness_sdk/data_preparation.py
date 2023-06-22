import os
from typing import Tuple, Dict, List
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.utils import to_categorical
import numpy as np


def RAFDB_subgroups_init() -> Tuple[Dict[str, List], Dict[str, List], Dict[str, List], Dict[str, List]]:
    """
    Initialize and return empty dictionaries for grouping images and labels by gender and race.

    Returns:
        A tuple of dictionaries with the following keys:
        - images_gender: a dictionary with keys 'male' and 'female'
        - labels_gender: a dictionary with keys 'male' and 'female'
        - images_race: a dictionary with keys 'asian', 'caucasian', and 'african'
        - labels_race: a dictionary with keys 'asian', 'caucasian', and 'african'
    """

    images_gender = {'male': [], 'female': []}
    labels_gender = {'male': [], 'female': []}

    images_race = {'asian': [], 'caucasian': [], 'african': []}
    labels_race = {'asian': [], 'caucasian': [], 'african': []}
    
    return images_gender, labels_gender, images_race, labels_race
    
def RAFDB_R50_preprocessing(features_path: str, filename: str, img_rows: int = 224, img_cols: int = 224) -> np.ndarray:
    """
    Preprocesses an image using the ResNet50 input preprocessing.

    Args:
        features_path (str): Path to the image file to preprocess.
        img_rows (int): Height of the output image. Default is 224.
        img_cols (int): Width of the output image. Default is 224.

    Returns:
        numpy.ndarray: Preprocessed image as a numpy array.
    """
    img = load_img(os.path.join(features_path, filename), target_size=(img_rows, img_cols))
    x = img_to_array(img)
    
    return x
    
def RAFDB_loader_api(features_path: str, annotations_path: str, labels_path: str, train: bool = True) -> dict:
    """
    Load and preprocess data for the RAF-DB dataset.

    Args:
        features_path (str): Path to the directory containing image features.
        annotations_path (str): Path to the directory containing annotation files.
        labels_path (str): Path to the file containing emotion labels.
        train (bool, optional): If True, load training data, else load test data. Default is True.

    Returns:
        dict: A dictionary containing the preprocessed data, split by gender and race.
    """
    
    num_classes = 7
    
    images_gender, labels_gender, images_race, labels_race = RAFDB_subgroups_init()

    # Loop through image files
    for filename in os.listdir(features_path):
        if train:
            status = 'train_'
        else:
            status = 'test_'

        if filename.startswith(status):
            is_valid = True
        else:
            continue

        # Extract image data (features)
        x = RAFDB_R50_preprocessing(features_path, filename)

        # Extract annotation data (labels)
        annotation_filename = filename.replace('.jpg', '_manu_attri.txt').replace('aligned_','')

        with open(f'{annotations_path}{annotation_filename}') as f:
            lines = f.readlines()
            gender = int(lines[5].strip())
            race = int(lines[6].strip())
            age = int(lines[7].strip())

        # Extract emotion label
        with open(labels_path) as f:
            for line in f:
                if line.startswith(filename.replace('_aligned','')):
                    emotion_label = int(line.split()[1]) - 1
                    break

        if is_valid:
            if gender == 0:
                images_gender['male'].append(x)
                labels_gender['male'].append(to_categorical(emotion_label, num_classes=num_classes))
            else:
                images_gender['female'].append(x)
                labels_gender['female'].append(to_categorical(emotion_label, num_classes=num_classes))

            if race == 0:
                images_race['caucasian'].append(x)
                labels_race['caucasian'].append(to_categorical(emotion_label, num_classes=num_classes))
            elif race == 1:
                images_race['asian'].append(x)
                labels_race['asian'].append(to_categorical(emotion_label, num_classes=num_classes))
            else:
                images_race['african'].append(x)
                labels_race['african'].append(to_categorical(emotion_label, num_classes=num_classes))

    # Convert lists to numpy arrays
    for gender in ['male', 'female']:
        images_gender[gender] = np.array(images_gender[gender])
        labels_gender[gender] = np.array(labels_gender[gender])
        
    for race in ['caucasian', 'african', 'asian']:
        images_race[race] = np.array(images_race[race])
        labels_race[race] = np.array(labels_race[race])

    raf_db = {'images_gender': images_gender, 'images_race': images_race,
              'labels_gender': labels_gender, 'labels_race': labels_race}

    return raf_db
