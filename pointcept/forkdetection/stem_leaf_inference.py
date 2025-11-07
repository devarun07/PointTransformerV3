import sys
sys.path.append('/home/admin_2qdjwp3/Arun/PointTransformerV3')

from pointcept.models.point_transformer_v3.point_transformer_v3m1_base import *
from torch.utils.data import DataLoader
import torch
from torch.nn import CrossEntropyLoss
from pointtransformer_architecture import PointTransformerV3
from forkdataset import ForkDataset, custom_collate_fn_steam_leaf_classifier
import os
import laspy
import numpy as np


MODEL_PATH = "/home/admin_2qdjwp3/Arun/PointTransformerV3/checkpoints/best_model_miou.pth"
TEST_DATA_PATH = "/home/admin_2qdjwp3/Arun/stem_leaf_classifier_dataset_LassiData_test.pkl"
PRED_SAVE_PATH = "/home/admin_2qdjwp3/Arun/PredictionData/"

BATCH_SIZE = 1          # inference runs 1 sample at a time
n_features = 4          # xyz + intensity
num_classes = 2         # stem, leaf


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running inference on: {device}")

# Load model
model = PointTransformerV3(in_channels=n_features, enable_flash=False)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

# Dataset & loader (no DistributedSampler)
test_dataset = ForkDataset(TEST_DATA_PATH, mode="test")
test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=custom_collate_fn_steam_leaf_classifier
)


# --------------------- INFERENCE LOOP --------------------- #
with torch.no_grad():
    for i, (batch, labels) in enumerate(test_loader):

        # Move tensors to GPU/CPU
        data_dict = {key: value.to(device) for key, value in batch.items()}
        data_dict["coord"] = data_dict["coord"].view(-1, 3)
        data_dict["feat"] = data_dict["feat"].view(-1, n_features)

        # Run model
        output_dict = model(data_dict)
        logits = output_dict["logits"]
        #logits = output_dict["feat"]                          # (N_points, num_classes)
        preds = torch.argmax(logits, dim=1).cpu().numpy()     # convert to numpy

        print(preds.shape)

        print(f"\n Inference result for sample #{i}:")
        print(f"Predicted class counts: {torch.bincount(torch.tensor(preds))}")

        coords = data_dict["coord"].cpu().numpy()
        labels = labels.cpu().numpy()

        # Save as laspy
        header = laspy.LasHeader(point_format=3, version="1.4")

        header.x_scale = 0.001
        header.y_scale = 0.001
        header.z_scale = 0.001

        header.x_offset = float(coords[:, 0].min())
        header.y_offset = float(coords[:, 1].min())
        header.z_offset = float(coords[:, 2].min())

        las = laspy.LasData(header)
        las.x = coords[:, 0]
        las.y = coords[:, 1]
        las.z = coords[:, 2]
        las.classification = labels

        las.add_extra_dim(
            laspy.ExtraBytesParams(
                name="pred",
                type=np.uint16,
                description="PointTransformer prediction"
            )
        )
        las.pred = preds

        las.write(f"{PRED_SAVE_PATH}/{i}_tree.las")


