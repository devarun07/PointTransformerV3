
# Clarification regarding the feature dimensions
# https://github.com/Pointcept/Pointcept/issues/222

import sys
sys.path.append('/home/arun/PointClouds/Pointcept')

from pointcept.forkdetection.forkdataset import ForkDataset, custom_collate_fn_fork_classifier
from pointcept.models.point_transformer_v3.point_transformer_v3m1_base import *
from torch.utils.data import DataLoader
import torch
from torch.nn import CrossEntropyLoss
from forkmodel import PointTransformerV3Classifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import wandb
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Training parameters
BATCH_SIZE = 2
NUM_EPOCHS = 100
LEARNING_RATE = 0.0001

# Wandb for visualization
wandb.init(
    project="tree-fork-classification", 
    config={
        "epochs": NUM_EPOCHS,
        "batch_size": BATCH_SIZE,
        "model": "PointTransformerV3"
    }
)

# Initialize the model
n_features = 4 # Only starting with xyz + intensity
model = PointTransformerV3Classifier(in_channels=n_features, 
                                 enable_flash=False,
                                 num_classes=1, # Binary classification
                                 cls_mode=True).to(device)

train_dataset = ForkDataset(dataset_path="/home/arun/PointClouds/Pointcept/fork_classifier_train.pkl")
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, collate_fn=custom_collate_fn_fork_classifier, shuffle=True, drop_last=True)

val_dataset = ForkDataset(dataset_path="/home/arun/PointClouds/Pointcept/fork_classifier_val.pkl", mode="eval")
val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, collate_fn=custom_collate_fn_fork_classifier, shuffle=True, drop_last=True)

best_model_savepath = 'fork_classifier.pt'

# Optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=40)
# criterion = CrossEntropyLoss()
criterion = torch.nn.BCEWithLogitsLoss() # Single class
EARLY_STOP = 50 # The number of train epochs after which training stops if no improvement


best_epoch = 0
best_val_accuracy = 0

# Training loop
for epoch in range(NUM_EPOCHS):
    train_loss_arr = []
    val_loss_arr = []

    running_loss = 0.0
    correct, total = 0, 0

    model.train()

    for i, (batch, labels) in enumerate(train_loader):
        data_dict = {key: value.to(device) for key, value in batch.items()}

        # Making sure the shapes are as expected
        data_dict['coord'] = data_dict['coord'].view(-1, 3)
        data_dict['feat'] = data_dict['feat'].view(-1, n_features)

        # Encoder output
        output_logits = model(data_dict).squeeze(dim=1)

        # Extract logics from the output dict
        logits = output_logits.to(device)
        labels = labels.to(device).float()

        # Compute loss
        loss = criterion(logits, labels)

        total_loss = loss.item()

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Compute loss and metrcis
        # preds = logits.argmax(dim=1)
        # correct += (preds == labels).sum().item()
        # total += labels.size(0)

        # Binary classification
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).long()
        correct += (preds == labels).sum().item()
        total += labels.size(0)


    # Update the learning rate
    scheduler.step()

    avg_loss = total_loss / len(train_loader)
    train_loss_arr.append(avg_loss)
    acc = correct / total
    print("Mean loss value for the epoch is {}".format(avg_loss))
    wandb.log({
        "epoch": epoch,
        "train_loss": avg_loss,
        "train_accuracy": acc
    })

    # Evalution mode
    model.eval()
    # Prediction arrays
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for i, (batch, labels) in enumerate(val_dataloader):
            data_dict = {key: value.to(device) for key, value in batch.items()}

            # Ensure the dimensions are consistent
            data_dict['coord'] = data_dict['coord'].view(-1, 3)
            data_dict['feat'] = data_dict['feat'].view(-1, n_features)

             # Encoder output
            output_logits = model(data_dict).squeeze(dim=1)

            # Extract logics from the output dict
            logits = output_logits.to(device)
            labels = labels.to(device).float()

            # Compute loss
            loss = criterion(logits, labels)

            total_val_loss = loss.item()
            val_loss_arr.append(total_val_loss)

            # preds = logits.argmax(dim=1)
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).long()
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())

    # Concatenate all predictions and labels
    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()

    # Compute confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    tn, fp, fn, tp = cm.ravel().tolist()
    val_accuracy = (tp + tn)/(tp + tn + fp + fn)

    print(f"\nConfusion Matrix (Epoch {epoch}):\n{cm}")

    wandb.log({
    "val_loss": np.mean(val_loss_arr),
    "confusion_matrix": wandb.plot.confusion_matrix(
        probs=None,
        y_true=all_labels,
        preds=all_preds,
        class_names=["No Fork", "Fork"] 
        ),
        "val_accuracy": val_accuracy
    })

    print("Epoch {} is finished!!".format(epoch))

    if val_accuracy <= best_val_accuracy:
        print(" ")
        
    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        best_epoch = epoch
        torch.save(model.state_dict(), best_model_savepath)
        print(f'Saved the best model at epoch {epoch+1}', flush=True)
    elif epoch - best_epoch > EARLY_STOP:
        print(f'Early stopping at epoch {epoch}, best validation accuracy {best_val_accuracy:.4f}', flush=True)
        break
