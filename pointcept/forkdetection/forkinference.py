import sys
sys.path.append('/home/arun/PointClouds/Pointcept')

from pointcept.forkdetection.forkdataset import ForkDataset, custom_collate_fn_fork_classifier
from pointcept.models.point_transformer_v3.point_transformer_v3m1_base import *
from torch.utils.data import DataLoader
import torch
from forkmodel import PointTransformerV3Classifier
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt


class ForkInference:
    def __init__(self, model_path='/home/arun/PointClouds/Pointcept/fork_classifier.pt'):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = PointTransformerV3Classifier(in_channels=4, 
                                 enable_flash=False,
                                 num_classes=2,
                                 cls_mode=True).to(self.device)
        self.model.load_state_dict(torch.load(model_path, weights_only=False))
        
        self.batch_size = 2
        self.test_dataset = ForkDataset(dataset_path="/home/arun/PointClouds/Pointcept/fork_classifier_val.pkl", mode="test")
        self.test_dataloader = DataLoader(self.test_dataset, batch_size=self.batch_size, collate_fn=custom_collate_fn_fork_classifier, shuffle=True, drop_last=True)
        self.all_preds, self.all_labels = [], []
        self.num_classes = 2

    def predict_fork_dataset(self):
        with torch.no_grad():
            for _, (batch, labels) in enumerate(self.test_dataloader):
                data_dict = {key: value.to(self.device) for key, value in batch.items()}

                # Ensure the dimensions are consistent
                data_dict['coord'] = data_dict['coord']
                data_dict['feat'] = data_dict['feat']

                # Encoder output
                output_logits = self.model(data_dict)

                # Extract logics from the output dict
                logits = output_logits.to(self.device)
                labels = labels.to(self.device)

                preds = logits.argmax(dim=1)
                self.all_preds.append(preds.cpu())
                self.all_labels.append(labels.cpu())

        # Concatenate all predictions and labels
        self.all_preds = torch.cat(self.all_preds).numpy()
        self.all_labels = torch.cat(self.all_labels).numpy()

        # Compute confusion matrix
        cm = confusion_matrix(self.all_labels, self.all_preds)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=["No Fork", "Fork"])
        disp.plot()
        plt.show()



if __name__ == "__main__":
    fork_inference = ForkInference()
    fork_inference.predict_fork_dataset()