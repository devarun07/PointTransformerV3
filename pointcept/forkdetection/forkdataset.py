import torch
from torch.utils.data import Dataset, DataLoader
import os
import laspy
import numpy as np
from pathlib import Path
import pickle
from sklearn.model_selection import train_test_split

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def rotate_around_z(points, angle_rad):
    rot_matrix = np.array([
        [np.cos(angle_rad), -np.sin(angle_rad), 0],
        [np.sin(angle_rad),  np.cos(angle_rad), 0],
        [0,                 0,                  1]
    ])
    xyz = points[:, :3] @ rot_matrix.T
    return np.hstack((xyz, points[:, 3:]))

def random_translate(points, max_translate=0.1):
    translation = np.random.uniform(-max_translate, max_translate, size=(1, 3))
    points[:, :3] += translation
    return points



def create_fork_labelled_dataset():
    true_files_path = '/home/arun/PointClouds/forkDetection/Dataset/forked_stemclassified_results'
    false_files_path = '/home/arun/PointClouds/forkDetection/Dataset/unforked_stemclassified_results'
    
    data = []
    labels = []

    for label, path in [(1, true_files_path), (0, false_files_path)]:
        for folder in list(Path(path).iterdir()):
            file = Path(os.path.join(folder, os.listdir(folder)[0]))
            las_file = laspy.read(file)
            trunk_ids = las_file.pred == 1

            points = np.vstack((las_file.x, las_file.y, las_file.z, las_file.intensity)).T[trunk_ids]
            points_mean = np.mean(points, axis=0)
            points = points - points_mean
            data.append(points)
            labels.append(label)

    # Split the dataset into train and validation sets
    val_ratio = 0.2
    train_data, val_data, train_labels, val_labels = train_test_split(
        data, labels, test_size=val_ratio, stratify=labels, random_state=123
    )

    # Augment training data
    augment_copies=2
    augmented_train_data = []
    augmented_train_labels = []

    for points, label in zip(train_data, train_labels):
        # Original sample
        augmented_train_data.append(points)
        augmented_train_labels.append(label)

        # Augmented copies
        for _ in range(augment_copies):
            angle_rad = np.random.uniform(-np.pi, np.pi)
            rotated = rotate_around_z(points, angle_rad)
            translated = random_translate(rotated)
            augmented_train_data.append(translated)
            augmented_train_labels.append(label)

    print(f"Original train samples: {len(train_data)}")
    print(f"Augmented total train samples: {len(augmented_train_data)}")

    # Save train data
    train_save_path = "/home/arun/PointClouds/Pointcept/fork_classifier_train.pkl"
    with open(train_save_path, 'wb') as f:
        pickle.dump({'data': augmented_train_data, 'labels': augmented_train_labels}, f)
    print(f"Saved {len(augmented_train_data)} training samples to {len(augmented_train_labels)}")

    # Save val data
    val_save_path = "/home/arun/PointClouds/Pointcept/fork_classifier_val.pkl"
    with open(val_save_path, 'wb') as f:
        pickle.dump({'data': val_data, 'labels': val_labels}, f)
    print(f"Saved {len(val_data)} validation samples to {val_save_path}")



def create_stem_leaf_dataset():
    true_files_path = '/home/arun/PointClouds/forkDetection/Dataset/forked_stemclassified_results'
    false_files_path = '/home/arun/PointClouds/forkDetection/Dataset/unforked_stemclassified_results'
    
    data = []
    labels = []

    for _, path in [(1, true_files_path), (0, false_files_path)]:
        for folder in list(Path(path).iterdir()):
            file = Path(os.path.join(folder, os.listdir(folder)[0]))
            las_file = laspy.read(file)
            trunk_ids = las_file.pred == 1
            points = np.vstack((las_file.x, las_file.y, las_file.z, las_file.intensity)).T[trunk_ids]
            # points = np.vstack((las_file.x, las_file.y, las_file.z, las_file.intensity)).T
            points_mean = np.mean(points, axis=0)
            points = points - points_mean
            data.append(points)
            labels.append(las_file.pred[trunk_ids])

    # Save the loaded data pickle
    save_path = "/home/arun/PointClouds/Pointcept/stem_leaf_classifier_dataset.pkl"
    with open(save_path, 'wb') as f:
        pickle.dump({'data': data, 'labels': labels}, f)

    print(f"Saved {len(data)} samples to {save_path}")


# Dataset for loading the forked tree dataset
class ForkDataset(Dataset):
    def __init__(
            self,
            dataset_path='/home/arun/PointClouds/Pointcept/fork_classifier_dataset.pkl',
            mode="train"
    ):
        super().__init__()
        self.data, self.labels = self.load_dataset(dataset_path)
        self.mode = mode


    def load_dataset(self, dataset_path):
        with open(dataset_path, "rb") as f:
            dataset = pickle.load(f)
            return dataset["data"], dataset["labels"]
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        points = torch.tensor(self.data[index], dtype=torch.float32)
        labels = torch.tensor(self.labels[index], dtype=torch.long) 

        coords = points[:, :3]
        features = points[:, :]

        if self.mode == 'train':
            coords, features = self.apply_augmentation(coords, features)

        data_dict = {
            'coord': coords,
            'feat': features
        }

        return data_dict, labels
    
    def apply_augmentation(self, coords, features):
        # Apply Gaussian noise to the points
        bbox_diag = np.linalg.norm(np.ptp(coords.numpy(), axis=0))
        coord_noise_std = 0.005 * bbox_diag
        coord_noise = torch.randn_like(coords) * coord_noise_std
        coords += coord_noise

        if features.shape[1] > 0:
            feat_range = torch.max(features, dim=0)[0] - torch.min(features, dim=0)[0]
            feat_noise_std = 0.001 * feat_range  # smaller than coord noise
            feat_noise = torch.randn_like(features) * feat_noise_std
            features += feat_noise

        return coords, features



def custom_collate_fn_steam_leaf_classifier(batch):
    batch_dict = {}
    labels_list = []
    offset_list = []
    batch_indices = []

    current_offset = 0

    # Pre-populate keys based on first sample
    keys = batch[0][0].keys()
    for key in keys:
        batch_dict[key] = []

    # Process each sample in the batch
    for i, (data_dict, label) in enumerate(batch):
        labels_list.append(label)
        offset_list.append(current_offset + len(label))
        current_offset += len(label)

        for key in keys:
            batch_dict[key].append(data_dict[key])

        batch_indices.append(
            torch.full((data_dict["coord"].shape[0],), i, dtype=torch.long)
        )

    # Merge point-wise data
    for key in keys:
        batch_dict[key] = torch.cat(batch_dict[key], dim=0)

    # Add 'batch' and 'offset' fields
    batch_dict["batch"] = torch.cat(batch_indices, dim=0)
    labels_out = torch.cat(labels_list, dim=0)

    device = labels_out.device
    batch_dict["offset"] = torch.tensor(offset_list, device=device, dtype=torch.long)
    batch_dict["grid_size"] = torch.tensor([0.01], device=device, dtype=torch.float32)

    return batch_dict, labels_out



def custom_collate_fn_fork_classifier(batch):
    batch_dict = {}
    labels_list = []
    offset_list = []
    batch_indices = []

    current_offset = 0

    # Pre-populate keys based on first sample
    keys = batch[0][0].keys()
    for key in keys:
        batch_dict[key] = []

    # Process each sample in the batch
    for i, (data_dict, label) in enumerate(batch):
        labels_list.append(label.unsqueeze(0))
        offset_list.append(current_offset + len(data_dict['coord']))
        current_offset += len(data_dict['coord'])

        for key in keys:
            batch_dict[key].append(data_dict[key])

        batch_indices.append(
            torch.full((data_dict["coord"].shape[0],), i, dtype=torch.long)
        )

    # Merge point-wise data
    for key in keys:
        batch_dict[key] = torch.cat(batch_dict[key], dim=0)

    # Add 'batch' and 'offset' fields
    batch_dict["batch"] = torch.cat(batch_indices, dim=0)
    labels_out = torch.cat(labels_list, dim=0)

    device = labels_out.device
    batch_dict["offset"] = torch.tensor(offset_list, device=device, dtype=torch.long)
    batch_dict["grid_size"] = torch.tensor([0.01], device=device, dtype=torch.float32)

    return batch_dict, labels_out



if __name__ == "__main__":
    
    # Test code to create the train data pickle
    create_fork_labelled_dataset()
    pass



