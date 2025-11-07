import sys
sys.path.append('/home/admin_2qdjwp3/Arun/PointTransformerV3')

from pointcept.models.point_transformer_v3.point_transformer_v3m1_base import *
from torch.utils.data import DataLoader
import torch
from torch.nn import CrossEntropyLoss
from pointtransformer_architecture import PointTransformerV3
from forkdataset import ForkDataset, custom_collate_fn_steam_leaf_classifier
import os


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Training parameters
BATCH_SIZE = 2
NUM_EPOCHS = 50
LEARNING_RATE = 0.0001
MODEL_SAVE_PATH = "/home/admin_2qdjwp3/Arun/PointTransformerV3/scripts/checkpoints"


os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

# Initialize the model
n_features = 4  # xyz + intensity
model = PointTransformerV3(in_channels=n_features, enable_flash=False).to(device)


#  Load TRAIN and TEST pickle datasets
train_dataset = ForkDataset("/home/admin_2qdjwp3/Arun/stem_leaf_classifier_dataset_LassiData_train.pkl", mode="train")
test_dataset  = ForkDataset("/home/admin_2qdjwp3/Arun/stem_leaf_classifier_dataset_LassiData_test.pkl", mode="test")

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                          collate_fn=custom_collate_fn_steam_leaf_classifier)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                         collate_fn=custom_collate_fn_steam_leaf_classifier)

# Optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=40)
criterion = CrossEntropyLoss()

best_val_loss = float("inf")

#  TRAINING LOOP
for epoch in range(NUM_EPOCHS):

    model.train()
    running_loss = 0.0

    for i, (batch, labels) in enumerate(train_loader):
        data_dict = {key: value.to(device) for key, value in batch.items()}
        data_dict["coord"] = data_dict["coord"].view(-1, 3)
        data_dict["feat"] = data_dict["feat"].view(-1, n_features)

        output_dict = model(data_dict)
        logits = output_dict["feat"].to(device)
        labels = labels.to(device)

        loss = criterion(logits, labels)
        running_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] | Batch [{i+1}/{len(train_loader)}] | Loss: {loss.item():.4f}")

    scheduler.step()
    train_loss = running_loss / len(train_loader)
    print(f"\n Epoch {epoch+1}/{NUM_EPOCHS} completed. Train Loss: {train_loss:.4f}")

    #  SAVE CHECKPOINT EVERY EPOCH
    torch.save(model.state_dict(), f"{MODEL_SAVE_PATH}/checkpoint_epoch_{epoch+1}.pth")

    #  Evaluate every 5 epochs
    if (epoch + 1) % 5 == 0:
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch, labels in test_loader:
                data_dict = {key: value.to(device) for key, value in batch.items()}
                data_dict["coord"] = data_dict["coord"].view(-1, 3)
                data_dict["feat"] = data_dict["feat"].view(-1, n_features)

                output_dict = model(data_dict)
                logits = output_dict["feat"].to(device)
                labels = labels.to(device)

                loss = criterion(logits, labels)
                val_loss += loss.item()

                predicted = torch.argmax(logits, dim=1)
                correct += (predicted == labels).sum().item()
                total += labels.numel()

        val_loss /= len(test_loader)
        accuracy = (correct / total) * 100

        print(f"TEST EPOCH {epoch+1}: Loss = {val_loss:.4f}, Accuracy = {accuracy:.2f}%")

        #  Save best model based on validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f"{MODEL_SAVE_PATH}/best_model.pth")
            print(f" Best model updated (val_loss: {best_val_loss:.4f})")

print("\n Training Completed!")
print(f" Models saved at: {MODEL_SAVE_PATH}")