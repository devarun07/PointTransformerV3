import laspy
from pathlib import Path
import os
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from jakteristics import las_utils, compute_features, FEATURE_NAMES
from sklearn.neighbors import KDTree
import fpsample


eigen_feature_names_all = ['eigenvalue_sum', 'omnivariance', 'eigenentropy', 'anisotropy', 'planarity',
                        'linearity', 'PCA1', 'PCA2', 'surface_variation', 'sphericity', 'verticality',
                          'nx', 'ny', 'nz', 'number_of_neighbors', 'eigenvalue1', 'eigenvalue2', 'eigenvalue3',
                            'eigenvector1x', 'eigenvector1y', 'eigenvector1z', 'eigenvector2x', 'eigenvector2y',
                              'eigenvector2z', 'eigenvector3x', 'eigenvector3y', 'eigenvector3z']

eigen_feature_names_selected = ['anisotropy', 'planarity',
                        'linearity', 'PCA1', 'sphericity', 'verticality']


def voxel_downsample(points, voxel_size):
    """points: Nx3 XYZ array"""
    # Compute voxel index for each point
    voxel_indices = np.floor(points / voxel_size).astype(np.int32)
    # Unique rows returns one point per voxel
    _, unique_idx = np.unique(voxel_indices, axis=0, return_index=True)
    return unique_idx


def create_dataset(feature_augmentation=False, downsampling=True):
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

            # Subtract only XYZ mean
            xyz = np.vstack((tree_points.x, tree_points.y, tree_points.z)).T
            xyz_mean = np.mean(xyz, axis=0)
            xyz = xyz - xyz_mean

            points = np.column_stack((xyz, tree_points.intensity))

            print(f"Original size: {xyz.shape}")

            if downsampling:
                VOXEL_SIZE = 0.07        # 7 cm voxel reduces leaf density
                MAX_POINTS = 15000       # Limit for PointTransformer, not exact target

                # 1) Voxel downsampling (structure-aware)
                voxel_idx = voxel_downsample(xyz, voxel_size=VOXEL_SIZE)
                xyz = xyz[voxel_idx]
                intensity = tree_points.intensity[voxel_idx]
                labels = las.classification[tree_mask][voxel_idx]
                points = np.column_stack((xyz, intensity))

                print(f"After voxel: {xyz.shape}")

                # 2) Only apply FPS if too big
                if xyz.shape[0] > MAX_POINTS:
                    np.random.seed(42)
                    fps_idx = fpsample.bucket_fps_kdtree_sampling(xyz, MAX_POINTS)
                    xyz = xyz[fps_idx]
                    intensity = intensity[fps_idx]
                    labels = labels[fps_idx]
                    points = points[fps_idx]
                    print(f"After FPS: {xyz.shape}\n")

            if feature_augmentation:

                # Add eigen based features
                features = compute_features(xyz, search_radius=0.35, feature_names=eigen_feature_names_selected, max_k_neighbors=20)
                points = np.hstack((points, features))

                if np.isnan(features).any() or np.isinf(features).any():
                    print("❌ Feature computation produced NaN/Inf!")

            train_data.append(points)
            train_labels.append(labels)

        
        # Test data
        for tree_id in np.unique(test):
            if tree_id == 0:
                continue # Skip the ground data
            
            tree_mask = tree_ids == tree_id
            tree_points = las.points[tree_mask]

            #tree_stem_mask = tree_mask & (classification == 1)

            # Subtract only XYZ mean
            xyz = np.vstack((tree_points.x, tree_points.y, tree_points.z)).T
            xyz_mean = np.mean(xyz, axis=0)
            xyz = xyz - xyz_mean

            points = np.column_stack((xyz, tree_points.intensity))
            labels = las.classification[tree_mask]


            if feature_augmentation:

                # Add eigen based features
                features = compute_features(xyz, search_radius=0.35, feature_names=eigen_feature_names_selected, max_k_neighbors=20)
                points = np.hstack((points, features))

                if np.isnan(features).any() or np.isinf(features).any():
                    print("❌ Feature computation produced NaN/Inf!")

            test_data.append(points)
            test_labels.append(labels)

    
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