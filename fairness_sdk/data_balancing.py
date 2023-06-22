import numpy as np


def RAFDB_train_balancing(data: dict) -> dict:
    """
    This function balances the train dataset by oversampling and undersampling data in different subgroups. 
    It takes a dictionary containing the train dataset and returns a dictionary containing the balanced dataset.

    Args:
        data (dict): A dictionary containing the train dataset with images and labels for gender and race subgroups.

    Returns:
        dict: A dictionary containing the balanced dataset with images and labels for gender and race subgroups.
    """
    
    train_images_gender, train_labels_gender, train_images_race, train_labels_race = initialize_subgroups()
    
    train_images_gender['male'] = np.array(data['images_gender']['male'])
    train_labels_gender['male'] = np.array(data['labels_gender']['male'])
    train_images_gender['female'] = np.array(data['images_gender']['female'])
    train_labels_gender['female'] = np.array(data['labels_gender']['female'])
    
    train_images_race['caucasian'] = np.array(data['images_race']['caucasian'])
    train_labels_race['caucasian'] = np.array(data['labels_race']['caucasian'])
    train_images_race['asian'] = np.array(data['images_race']['asian'])
    train_labels_race['asian'] = np.array(data['labels_race']['asian'])
    train_images_race['african'] = np.array(data['images_race']['african'])
    train_labels_race['african'] = np.array(data['labels_race']['african'])
    
    max_samples = max(
        len(train_images_gender['male']),
        len(train_images_gender['female']),
        len(train_images_race['asian']),
        len(train_images_race['caucasian']),
        len(train_images_race['african'])
    )
    
    for gender in ['male', 'female']:
        if len(train_images_gender[gender]) > max_samples:
            indices = np.random.choice(len(train_images_gender[gender]), max_samples, replace=False)
            train_images_gender[gender] = train_images_gender[gender][indices]
            train_labels_gender[gender] = train_labels_gender[gender][indices]
            
        elif len(train_images_gender[gender]) < max_samples:
            num_samples = max_samples - len(train_images_gender[gender])
            indices = np.random.choice(len(train_images_gender[gender]), num_samples, replace=True)
            train_images_gender[gender] = np.concatenate((train_images_gender[gender], train_images_gender[gender][indices]))
            train_labels_gender[gender] = np.concatenate((train_labels_gender[gender], train_labels_gender[gender][indices]))

    for race in train_images_race.keys():
        num_samples = len(train_images_race[race])
        
        if num_samples < max_samples:
            num_samples_to_add = max_samples - num_samples
            random_indices = np.random.choice(num_samples, num_samples_to_add)
            train_images_race[race] = np.concatenate((train_images_race[race], train_images_race[race][random_indices]), axis=0)
            train_labels_race[race] = np.concatenate((train_labels_race[race], train_labels_race[race][random_indices]), axis=0)
    
    balanced_raf_db = {'images_gender': train_images_gender, 'images_race': train_images_race,
                       'labels_gender': train_labels_gender, 'labels_race': train_labels_race}
    
    return balanced_raf_db

def RAFDB_test_balancing(test_images_gender: np.ndarray, test_labels_gender: np.ndarray, test_images_race: np.ndarray, test_labels_race: np.ndarray, num_samples: int) -> np.ndarray:
        """
        Balance the test data for gender and race subgroups by oversampling.

        Args:
            test_images_gender (np.ndarray): Test images for gender subgroups.
            test_labels_gender (np.ndarray): Test labels for gender subgroups.
            test_images_race (np.ndarray): Test images for race subgroups.
            test_labels_race (np.ndarray): Test labels for race subgroups.
            num_samples (int): Number of samples to generate for each subgroup.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: A tuple containing the oversampled test images and labels for gender and race subgroups.
        """
    # Initialize empty lists for the oversampled test data and labels
    test_images_gender_oversampled = {'male': [], 'female': []}
    test_labels_gender_oversampled = {'male': [], 'female': []}

    test_images_race_oversampled = {'asian': [], 'caucasian': [], 'african': []}
    test_labels_race_oversampled = {'asian': [], 'caucasian': [], 'african': []}

    # Loop through the test data for gender subgroups
    for gender in ['male', 'female']:
        # Get the number of samples in the original test data for this subgroup
        num_samples_gender = len(test_images_gender[gender])

        # Calculate the number of additional samples needed for this subgroup
        num_samples_needed = num_samples - num_samples_gender

        # Randomly select additional samples from the original test data for this subgroup
        idx = random.choices(range(num_samples_gender), k=num_samples_needed)
        test_images_gender_oversampled[gender] = np.concatenate([test_images_gender[gender], test_images_gender[gender][idx]])
        test_labels_gender_oversampled[gender] = np.concatenate([test_labels_gender[gender], test_labels_gender[gender][idx]])

    # Loop through the test data for race subgroups
    for race in ['asian', 'caucasian', 'african']:
        # Get the number of samples in the original test data for this subgroup
        num_samples_race = len(test_images_race[race])

        # Calculate the number of additional samples needed for this subgroup
        num_samples_needed = num_samples - num_samples_race

        # Randomly select additional samples from the original test data for this subgroup
        idx = random.choices(range(num_samples_race), k=num_samples_needed)
        test_images_race_oversampled[race] = np.concatenate([test_images_race[race], test_images_race[race][idx]])
        test_labels_race_oversampled[race] = np.concatenate([test_labels_race[race], test_labels_race[race][idx]])

    for gender in ['male', 'female']:
        test_images_gender_oversampled[gender] = test_images_gender_oversampled[gender][:num_samples]
        test_labels_gender_oversampled[gender] = test_labels_gender_oversampled[gender][:num_samples]

    for race in ['african', 'asian', 'caucasian']:
        test_images_race_oversampled[race] = test_images_race_oversampled[race][:num_samples]
        test_labels_race_oversampled[race] = test_labels_race_oversampled[race][:num_samples]
       
    return test_images_gender_oversampled, test_labels_gender_oversampled, test_images_race_oversampled, test_labels_race_oversampled
