import laspy
from pathlib import Path
import os
import pickle
import numpy as np
from sklearn.model_selection import train_test_split


def create_dataset():
    filesPath = "/home/admin_2qdjwp3/Arun/LassiData/data/"

    train_data = []
    train_labels = []

    test_data = []
    test_labels = []

    for file in Path(filesPath).iterdir():
        las = laspy.read(file)

        # Better to train the model to identify stem and leaf from only individual tree data
        tree_ids = las.tree_index

        train, test = train_test_split(np.unique(tree_ids), test_size=0.2, random_state=123)

        # Train data
        for tree_id in np.unique(train):
            if tree_id == 0:
                continue # Skip the ground data
            
            tree_mask = tree_ids == tree_id
            tree_points = las.points[tree_mask]

            #tree_stem_mask = tree_mask & (classification == 1)

            points = np.vstack((tree_points.x, tree_points.y, tree_points.z, tree_points.intensity)).T
            points_mean = np.mean(points, axis=0)
            points = points - points_mean

            train_data.append(points)
            train_labels.append(tree_points.classification)

        
        # Test data
        for tree_id in np.unique(test):
            if tree_id == 0:
                continue # Skip the ground data
            
            tree_mask = tree_ids == tree_id
            tree_points = las.points[tree_mask]

            #tree_stem_mask = tree_mask & (classification == 1)

            points = np.vstack((tree_points.x, tree_points.y, tree_points.z, tree_points.intensity)).T
            points_mean = np.mean(points, axis=0)
            points = points - points_mean

            test_data.append(points)
            test_labels.append(tree_points.classification)

    
    # Save the train data
    save_path = "/home/admin_2qdjwp3/Arun/stem_leaf_classifier_dataset_LassiData_train.pkl"
    with open(save_path, 'wb') as f:
        pickle.dump({'data': train_data, 'labels': train_labels}, f)
        print(f"Saved {len(train_data)} samples to {save_path}")

    # Save the test data
    save_path = "/home/admin_2qdjwp3/Arun/stem_leaf_classifier_dataset_LassiData_test.pkl"
    with open(save_path, 'wb') as f:
        pickle.dump({'data': test_data, 'labels': test_labels}, f)
        print(f"Saved {len(test_data)} samples to {save_path}")



if __name__ == "__main__":
    create_dataset()