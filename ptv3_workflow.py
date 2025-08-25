import torch
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Dataset, DataLoader
from sklearn.decomposition import PCA

from torch.optim import AdamW
import torch.nn as nn

import os
import sys
from datetime import datetime

pointcept_path = '/projappl/project_2013395/Pointcept_tlukkari'
sys.path.append(pointcept_path)

from pointcept.models.utils.losses import *
from pointcept.models.utils.augmentation import *
from pointcept.models.utils.metrics import *
from pointcept.models.utils.utils import *
from pointcept.models.point_transformer_v3.point_transformer_v3m1_base import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

path_to_road_segments = '/scratch/project_2013395/data_train_val_test'
train_fpaths, validation_fpaths, test_fpaths = fetch_train_val_test_sets(folder_path=path_to_road_segments, seed=42, proportion=0.38)

# This is how the data samples are constructed in the dataset:
# index: attribute
# 0: X
# 1: Y
# 2: Z
# 3: intensity
# 4: classification
# 5: label 
class RoadSegmentDataset(Dataset):
    def __init__(self, sample_paths=[], Npoints=30000, transform=None, pca_preprocess=False, augmentation=None, mode="train", feature_mask=[0,1,2,3]):
        self.sample_paths = sample_paths            # List of paths to the sample files
        self.transform = transform                  # Transform function to preprocess the points
        self.pca_preprocess = pca_preprocess        # Flag indicating whether to apply PCA preprocessing
        self.augmentation = augmentation            # Augmentation function to apply
        self.mode = mode                            # Mode of the dataset (e.g., 'train', 'test')
        self.feature_mask = feature_mask            # Indices of the features to be used from the point data
        self.Npoints = Npoints

    def __len__(self):
        return len(self.sample_paths)

    def __getitem__(self, idx):
        samplepath = self.sample_paths[idx]
        points = np.load(samplepath)

        points = points[points[:,4] == 2] # Consider only ground points

        npoints_orig = points.shape[0]              # Get the number of points in the original point cloud

        # Randomly downsample if there are over self.Npoint points
        if npoints_orig >= self.Npoints:
            indices = np.random.choice(npoints_orig, size=self.Npoints, replace=False)  # Randomly sample Npoints
            points = points[indices, :]
                
        if self.transform:
            # Preprocess the points.
            points = self.transform(points, use_pca=self.pca_preprocess)
            
        if self.augmentation:
            # Apply random augmentations to the points
            points = self.augmentation(points)

        # Extract the features specified by the feature mask
        features = torch.tensor(points[:, self.feature_mask], dtype=torch.float32)

        coords = torch.tensor(points[:, :3], dtype=torch.float32)
        
        data_dict = {'coord': coords,
                     #"original_coord": original_coordinates,
                     'feat': features,
                    }
        
        # Extract the labels of the points
        labels = torch.from_numpy(points[:,5])

        return data_dict, labels.to(torch.long)


def custom_collate_fn(batch):
    batch_dict = {}
    batch_dict["batch"] = []

    labels_list = []
    offset_list = []
    
    current_offset = 0

    # Iterate over each item in the batch
    for i, (data_dict, label) in enumerate(batch):
        
        # Append label to the labels list
        labels_list.append(label)
        
        # Add the current length of the label to the offset list
        offset_list.append(current_offset + len(label))
        current_offset += len(label)
        
        # Iterate over the keys in data_dict and append their values to the lists in batch_dict
        for key, value in data_dict.items():
            if key not in batch_dict:
                batch_dict[key] = []
            batch_dict[key].append(value)

        # Create and add batch tensor, that indicates to which point cloud the points belongs to
        segment_length = data_dict["coord"].shape[0]
        batch_tensor = torch.full((segment_length,), i, dtype=torch.long).view(-1)
        batch_dict["batch"].append(batch_tensor)

    # Convert the labels list to a tensor
    labels_out = torch.cat(labels_list, dim=0)  # Concatenate labels 

    # Concatenate all the values
    for key, value in batch_dict.items():
        batch_dict[key] = torch.cat(value, dim=0)

    # Add the offset to the batch_dict
    batch_dict['offset'] = torch.tensor(offset_list, device=labels_out.device, dtype=torch.long)

    batch_dict['grid_size'] = torch.tensor([0.01], device=labels_out.device, dtype=torch.float32)

    return batch_dict, labels_out

feature_mask = [0,1,2,3] # point attributes to use in training
n_features = len(feature_mask)
patch_size = 512 # define patch size for PTv3 model

# DEFAULT MODEL
"""model = PointTransformerV3(
    in_channels=n_features,  # Number of input features per point
    out_channels=3,
    order=("z", "z-trans", "hilbert", "hilbert-trans"),  # Order of serialization
    stride=(2, 2, 2, 2),  # Stride for each stage
    enc_depths=(2, 2, 2, 6, 2),  # Depth of the encoder at each stage
    enc_channels=(32, 64, 128, 256, 512),  # Number of channels in the encoder at each stage
    enc_num_head=(2, 4, 8, 16, 32),  # Number of attention heads in the encoder
    enc_patch_size=(patch_size, patch_size, patch_size, patch_size, patch_size),  # Patch size for the encoder
    dec_depths=(2, 2, 2, 2),  # Depth of the decoder at each stage
    dec_channels=(64, 64, 128, 256),  # Number of channels in the decoder at each stage
    dec_num_head=(4, 4, 8, 16),  # Number of attention heads in the decoder
    dec_patch_size=(patch_size, patch_size, patch_size, patch_size),  # Patch size for the decoder
    mlp_ratio=4,  # Ratio of MLP hidden dimension to embedding dimension
    qkv_bias=True,  # Bias in the query, key, value projections
    qk_scale=None,  # Scale factor for query and key projections
    attn_drop=0.0,  # Dropout rate for attention weights #0.0
    proj_drop=0.0,  # Dropout rate for projection weights #0.0
    drop_path=0.2,  # Dropout rate for stochastic depth #0.3
    pre_norm=True,  # Whether to apply normalization before attention and MLP blocks
    shuffle_orders=True,  # Whether to shuffle orders during serialization
    enable_rpe=True,  # Whether to enable relative positional encoding
    enable_flash=False,  # Whether to enable flash attention
    upcast_attention=False,  # Whether to use upcasting in attention
    upcast_softmax=False,  # Whether to use upcasting in softmax
    cls_mode=False,  # Whether to enable classification mode
    pdnorm_bn=False,  # Whether to use PDNorm with BatchNorm
    pdnorm_ln=False,  # Whether to use PDNorm with LayerNorm
    pdnorm_decouple=True,  # Whether to decouple PDNorm
    pdnorm_adaptive=False,  # Whether to use adaptive PDNorm
    pdnorm_affine=True,  # Whether to use affine transformation in PDNorm
    pdnorm_conditions=("ScanNet", "S3DIS", "Structured3D")  # Conditions for PDNorm
).to(device)"""

# LOW COMPLEXITY MODEL
model = PointTransformerV3(
        in_channels=n_features,
        out_channels=3,
        order=("z", "z-trans", "hilbert", "hilbert-trans"),
        stride=(2, 2), 
        enc_depths=(2, 2, 2),  
        enc_channels=(32, 64, 128),  
        enc_num_head=(2, 4, 8),  
        enc_patch_size=(patch_size, patch_size, patch_size), 
        dec_depths=(2, 2),  
        dec_channels=(32, 64),  
        dec_num_head=(2, 4),  
        dec_patch_size=(patch_size, patch_size),  
        mlp_ratio=2, 
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.1,  
        proj_drop=0.1,  
        drop_path=0.1,  
        pre_norm=True,
        shuffle_orders=True,
        enable_rpe=True,
        enable_flash=False,
        upcast_attention=False,
        upcast_softmax=False,
        cls_mode=False,
        pdnorm_bn=False,
        pdnorm_ln=False,
        pdnorm_decouple=True,
        pdnorm_adaptive=False,
        pdnorm_affine=True,
        pdnorm_conditions=("ScanNet", "S3DIS", "Structured3D"),
    ).to(device)

# MODERATE COMPLEXITY MODEL
"""model = PointTransformerV3(
    in_channels=n_features,  # Number of input features per point
    out_channels=3,  # Number of output classes
    order=("z", "z-trans", "hilbert", "hilbert-trans"),  
    stride=(2, 2, 2),  # Stride for each stage, reduced stages for moderate complexity
    enc_depths=(2, 2, 4, 2),  # Depth of the encoder at each stage, moderately deep
    enc_channels=(32, 64, 128, 256),  # Number of channels in the encoder at each stage, moderate number of channels
    enc_num_head=(2, 4, 8, 16),  # Number of attention heads in the encoder
    enc_patch_size=(patch_size, patch_size, patch_size, patch_size),  # Patch size for the encoder
    dec_depths=(2, 2, 2),  # Depth of the decoder at each stage
    dec_channels=(64, 128, 256),  # Number of channels in the decoder at each stage
    dec_num_head=(2, 4, 8),  # Number of attention heads in the decoder
    dec_patch_size=(patch_size, patch_size, patch_size),  # Patch size for the decoder
    mlp_ratio=4,  # Ratio of MLP hidden dimension to embedding dimension
    qkv_bias=True,  # Bias in the query, key, value projections
    qk_scale=None,  # Scale factor for query and key projections
    attn_drop=0.2,  # Dropout rate for attention weights
    proj_drop=0.2,  # Dropout rate for projection weights
    drop_path=0.2,  # Dropout rate for stochastic depth
    pre_norm=True,  # Whether to apply normalization before attention and MLP blocks
    shuffle_orders=True,  # Whether to shuffle orders during serialization
    enable_rpe=True,  # Whether to enable relative positional encoding
    enable_flash=False,  # Whether to enable flash attention
    upcast_attention=False,  # Whether to use upcasting in attention
    upcast_softmax=False,  # Whether to use upcasting in softmax
    cls_mode=False,  # Whether to enable classification mode
    pdnorm_bn=False,  # Whether to use PDNorm with BatchNorm
    pdnorm_ln=False,  # Whether to use PDNorm with LayerNorm
    pdnorm_decouple=True,  # Whether to decouple PDNorm
    pdnorm_adaptive=False,  # Whether to use adaptive PDNorm
    pdnorm_affine=True,  # Whether to use affine transformation in PDNorm
    pdnorm_conditions=("ScanNet", "S3DIS", "Structured3D"),  # Conditions for PDNorm
).to(device)"""

model_savefolder = '/projappl/project_2013395/Pointcept_tlukkari/pointcept/models/point_transformer_v3/saved_models'
best_model_savename = f"test_example.pth"
best_model_savepath = os.path.join(model_savefolder, best_model_savename)

NUM_EPOCHS = 100
BATCH_SIZE = 4
LEARNING_RATE = 0.0005

optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)#, weight_decay=0.00001)
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=40, T_mult=1)
criterion = get_loss("lovasz_loss")
EARLY_STOP_THRESHOLD = 50 # How many epochs to train before terminating if the validation mIoU does not improve

train_dataset = RoadSegmentDataset(sample_paths=train_fpaths, transform=preprocess_points, pca_preprocess=True, augmentation=augment_points, mode="train")
val_dataset = RoadSegmentDataset(sample_paths=validation_fpaths, transform=preprocess_points, pca_preprocess=True, mode="val")

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, collate_fn=custom_collate_fn, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, collate_fn=custom_collate_fn, shuffle=False)

# Lists to store loss and mean IoU
loss_values = []
loss_values_validation = []

miou_trains = []
miou_vals = []

miou_trains_0 = []
miou_trains_1 = []
miou_trains_2 = []


miou_vals_0 = []
miou_vals_1 = []
miou_vals_2 = []


best_miou_val = 0.0
best_epoch = 0

start = datetime.now()

# Training loop
for epoch in range(NUM_EPOCHS): 
    total_loss = 0
    total_loss_val = 0
    
    miou_train_epoch = []
    miou_val_epoch = []

    miou_train_epoch_0 = []
    miou_train_epoch_1 = []
    miou_train_epoch_2 = []


    miou_val_epoch_0 = []
    miou_val_epoch_1 = []
    miou_val_epoch_2 = []


    # Put the model into training mode
    model.train()

    for i, (batch, labels) in enumerate(train_loader):
        data_dict = {key: value.to(device) for key, value in batch.items()}
        labels = labels.to(device)
        labels = labels.unsqueeze(0)
        
        # Ensure the dimensions are consistent
        data_dict['coord'] = data_dict['coord'].view(-1, 3)
        data_dict['feat'] = data_dict['feat'].view(-1, n_features)

        # Forward pass
        output_dict = model(data_dict)

        # Extract logits from the output dict
        logits = output_dict['feat'].transpose(0, 1).unsqueeze(0)

        # Compute the loss
        loss = criterion(logits, labels)
        
        total_loss += loss.item()

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Compute metrics
        miou, per_class_ious = iou(logits, labels)
        
        # Collect metrics
        miou_train_epoch.append(miou)
        miou_train_epoch_0.append(per_class_ious[0])
        miou_train_epoch_1.append(per_class_ious[1])
        miou_train_epoch_2.append(per_class_ious[2])


    # Update the learning rate
    scheduler.step()
    
    # Compute the average loss
    avg_loss = total_loss / len(train_loader)

    # Store the loss
    loss_values.append(avg_loss)
    
    # Store mean IoU of the epoch
    miou_train = np.mean(miou_train_epoch)
    miou_trains.append(miou_train)

    # Store IoU per classes
    miou_train_0 = np.mean(miou_train_epoch_0)
    miou_trains_0.append(miou_train_0)

    miou_train_1 = np.mean(miou_train_epoch_1)
    miou_trains_1.append(miou_train_1)
    
    miou_train_2 = np.mean(miou_train_epoch_2)
    miou_trains_2.append(miou_train_2)

    #print(f'Epoch: [{epoch+1}/{NUM_EPOCHS}], train loss: {avg_loss:.4f}, train mIoU {miou_train:.4f}', end=" ")
    print(f'Epoch: [{epoch+1}/{NUM_EPOCHS}], train loss: {avg_loss:.4f}, train mIoU: {miou_train:.4f}, train IoU for class 0: {miou_train_0:.4f}, train IoU for class 1: {miou_train_1:.4f}, train IoU for class 2: {miou_train_2:.4f}', end=" ", flush=True)

    
    # Put the model into evaluation mode
    model.eval()
    
    with torch.no_grad():
        for i, (batch, labels) in enumerate(val_loader):
            data_dict = {key: value.to(device) for key, value in batch.items()}
            labels = labels.to(device).unsqueeze(0)

            # Ensure the dimensions are consistent
            data_dict['coord'] = data_dict['coord'].view(-1, 3)
            data_dict['feat'] = data_dict['feat'].view(-1, n_features)

            # Forward pass
            output_dict = model(data_dict)

            # Extract logits from the output dict
            logits = output_dict['feat'].transpose(0, 1).unsqueeze(0)
        
            # Compute the loss
            loss = criterion(logits, labels)
            
            total_loss_val += loss.item()
            
            # Compute metrics
            #overall_accuracy, per_class_accuracies = accuracy(logits, labels)
            miou, per_class_ious = iou(logits, labels)
            
            # Collect metrics
            miou_val_epoch.append(miou)
            miou_val_epoch_0.append(per_class_ious[0])
            miou_val_epoch_1.append(per_class_ious[1])
            miou_val_epoch_2.append(per_class_ious[2])

        # Compute the average loss 
        avg_loss_val = total_loss_val / len(val_loader)
        loss_values_validation.append(avg_loss_val)
        
        # Store mean IoU of the epoch
        miou_val = np.mean(miou_val_epoch)
        miou_vals.append(miou_val)

        miou_val_0 = np.mean(miou_val_epoch_0)
        miou_vals_0.append(miou_val_0)

        miou_val_1 = np.mean(miou_val_epoch_1)
        miou_vals_1.append(miou_val_1)

        miou_val_2 = np.mean(miou_val_epoch_2)
        miou_vals_2.append(miou_val_2)
        
        print(f'### val loss: {avg_loss_val:.4f}, val mIoU: {miou_val:.4f}, val IoU for class 0: {miou_val_0:.4f}, val IoU for class 1: {miou_val_1:.4f}, val IoU for class 2: {miou_val_2:.4f}', end=" ", flush=True)

    if miou_val <= best_miou_val:
        print(" ")
        
    if miou_val > best_miou_val:
        best_miou_val = miou_val
        best_epoch = epoch
        torch.save(model.state_dict(), best_model_savepath)
        print(f'Saved the best model at epoch {epoch+1}', flush=True)
    elif epoch - best_epoch > EARLY_STOP_THRESHOLD:
        print(f'Early stopping at epoch {epoch}, best validation mIoU {best_miou_val:.4f}', flush=True)
        break

print("Training completed in: " + str(datetime.now() - start))