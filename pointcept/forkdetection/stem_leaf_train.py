import sys
sys.path.append('/home/arun/PointClouds/Pointcept')

from pointcept.forkdetection.forkdataset import ForkDataset, custom_collate_fn_steam_leaf_classifier
from pointcept.models.point_transformer_v3.point_transformer_v3m1_base import *
from torch.utils.data import DataLoader
import torch
from torch.nn import CrossEntropyLoss
from pointtransformer_architecture import PointTransformerV3
import wandb


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Training parameters
BATCH_SIZE = 1
NUM_EPOCHS = 5
LEARNING_RATE = 0.0001

# Wandb for visualization
wandb.init(
    project="stem-leaf-segmentation", 
    config={
        "epochs": NUM_EPOCHS,
        "batch_size": BATCH_SIZE,
        "model": "PointTransformerV3"
    }
)

# Initialize the model
n_features = 4 # Only starting with xyz + intensity
model = PointTransformerV3(in_channels=n_features, 
                                 enable_flash=False).to(device)

train_dataset = ForkDataset(dataset_path="/home/arun/PointClouds/Pointcept/stem_leaf_classifier_dataset.pkl")
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, collate_fn=custom_collate_fn_steam_leaf_classifier, shuffle=True)

# Optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=40)
criterion = CrossEntropyLoss()
EARLY_STOP = 50 # The number of train epochs after which training stops if no improvement


# Training loop
for epoch in range(NUM_EPOCHS):

    model.train()

    for i, (batch, labels) in enumerate(train_loader):
        data_dict = {key: value.to(device) for key, value in batch.items()}

        # Making sure the shapes are as expected
        data_dict['coord'] = data_dict['coord'].view(-1, 3)
        data_dict['feat'] = data_dict['feat'].view(-1, n_features)

        # Encoder output
        output_dict = model(data_dict)

        # Extract logics from the output dict
        logits = output_dict['feat'].to(device)
        labels = labels.to(device)

        # Compute loss
        loss = criterion(logits, labels)

        total_loss = loss.item()

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


    # Update the learning rate
    scheduler.step()

    print("Epoch {} is finished!!".format(epoch))
