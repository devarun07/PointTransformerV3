import torch
from torch.utils.data import Dataset, DataLoader
import pickle



class StemLeafDataset(Dataset):
    def __init__(self, 
            dataset_path='/home/admin_2qdjwp3/Arun/stem_leaf_classifier_dataset_LassiData.pkl',
            mode="train"):
        
        super().__init__()
        self.dataset_path = dataset_path
        self.mode = mode
        self.data, self.labels = self.load_dataset()


    def load_dataset(self):
        with open(self.dataset_path, "rb") as f:
            pickle_data = pickle.load(f)

        return pickle_data['data'], pickle_data['labels']
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        points = torch.Tensor(self.data[index], dtype=torch.float32)
        label = torch.Tensor(self.labels[index], dtype=torch.long)

        if self.mode == "train":
            points = self.apply_augmentation(points)

        coords = points[:, :3]
        features = points[:, :]

        data_dict = {
            'coord': coords,
            'feat': features
        }
        
        return data_dict, label

    def apply_augmentation(self, points):
        # placeholder
        return points
    


dataset = StemLeafDataset()

        