import sys
sys.path.append('/home/admin_2qdjwp3/Arun/PointTransformerV3')

from pointcept.models.point_transformer_v3.point_transformer_v3m1_base import *
from torch.utils.data import DataLoader, DistributedSampler
import torch
from torch.nn import CrossEntropyLoss
from pointtransformer_architecture import PointTransformerV3
from forkdataset import ForkDataset, custom_collate_fn_steam_leaf_classifier
import os
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist


# Training parameters
BATCH_SIZE = 1
NUM_EPOCHS = 50
LEARNING_RATE = 0.0001
MODEL_SAVE_PATH = "/home/admin_2qdjwp3/Arun/PointTransformerV3/checkpoints"


def setup():
    "Initialize distributed training environment"
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))


def cleanup():
    dist.destroy_process_group()


def main():
    setup()

    # Get system information
    local_rank = int(os.environ["LOCAL_RANK"]) # gpu id
    world_size = dist.get_world_size() # num of gpus

    device = torch.device(f"cuda:{local_rank}")


    # Initialize the model
    n_features = 4  # xyz + intensity
    num_classes = 2
    model = PointTransformerV3(in_channels=n_features, num_classes=num_classes, enable_flash=False).to(device)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    #  Load TRAIN and TEST pickle datasets
    train_dataset = ForkDataset("/home/admin_2qdjwp3/Arun/stem_leaf_classifier_dataset_LassiData_train.pkl", mode="train")
    test_dataset  = ForkDataset("/home/admin_2qdjwp3/Arun/stem_leaf_classifier_dataset_LassiData_test.pkl", mode="test")

    train_sampler = DistributedSampler(train_dataset, shuffle=True) # Other values are picked automatically from the distributed group
    test_sampler = DistributedSampler(test_dataset, shuffle=False)

    train_loader = DataLoader(train_dataset, 
                            batch_size=BATCH_SIZE,
                            pin_memory=False,
                            num_workers=0,
                            sampler=train_sampler,
                            collate_fn=custom_collate_fn_steam_leaf_classifier)

    test_loader = DataLoader(test_dataset,
                            batch_size=BATCH_SIZE,
                            pin_memory=False,
                            num_workers=0,
                            sampler=test_sampler,
                            collate_fn=custom_collate_fn_steam_leaf_classifier)

    # optmization
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=30)
    criterion = CrossEntropyLoss()
    best_miou = 0.0


    # ------------------------------ Training Loop ------------------------------------------
    for epoch in range(NUM_EPOCHS):

        train_sampler.set_epoch(epoch) # This is for preventing different GPUs end up getting same data indices, randome state of every gpu should be same to get same shuffle

        model.train()
        running_loss = 0.0

        for i, (batch, labels) in enumerate(train_loader):
            data_dict = {key: value.to(device) for key, value in batch.items()}
            data_dict["coord"] = data_dict["coord"].view(-1, 3)
            data_dict["feat"] = data_dict["feat"].view(-1, n_features)

            output_dict = model(data_dict)
            logits = output_dict["logits"]
            labels = labels.to(device)

            loss = criterion(logits, labels)
            running_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if local_rank == 0:   ### <-- log only on rank 0
                print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] | Batch [{i+1}/{len(train_loader)}] | Loss: {loss.item():.4f}")

        scheduler.step()
        train_loss = running_loss/len(train_loader)

        if local_rank == 0:
            print(f"\n Epoch {epoch+1}/{NUM_EPOCHS} completed. Train Loss: {train_loss:.4f}")
            torch.save(model.module.state_dict(), f"{MODEL_SAVE_PATH}/checkpoint_epoch_{epoch+1}.pth")

        
        # Validation pipeline
        if (epoch + 1) % 5 == 0 and local_rank == 0:
            
            inter_per_class = torch.zeros(num_classes, dtype=torch.float64)
            union_per_class = torch.zeros(num_classes, dtype=torch.float64)

            model.eval()
            with torch.no_grad():
                for batch, labels in test_loader:

                    data_dict = {k: v.to(device) for k, v in batch.items()}
                    data_dict["coord"] = data_dict["coord"].view(-1, 3)
                    data_dict["feat"] = data_dict["feat"].view(-1, n_features)

                    
                    output_dict = model(data_dict)
                    logits = output_dict["logits"]
                    preds = torch.argmax(logits, dim=1)

                    preds = preds.cpu()
                    labels = labels.cpu()

                    for cls in range(num_classes):
                        pred_mask  = preds == cls
                        label_mask = labels == cls

                        inter = (pred_mask & label_mask).sum()
                        union = (pred_mask | label_mask).sum()

                        inter_per_class[cls] += inter
                        union_per_class[cls] += union

            # Final mIoU
            iou_per_class = inter_per_class / (union_per_class + 1e-6)
            miou = iou_per_class.mean()

            print(f" Validation Epoch {epoch+1}")
            print("IoU per class:", iou_per_class.numpy())
            print(f"mIoU: {miou.item():.4f}")

            # ---------- SAVE BEST MODEL BASED ON MIOU ----------
            if miou > best_miou:
                best_miou = miou
                torch.save(model.module.state_dict(), f"{MODEL_SAVE_PATH}/best_model_miou.pth")
                print(f" Best model updated (mIoU improved to {best_miou:.4f})")



if __name__ == "__main__":
    main()