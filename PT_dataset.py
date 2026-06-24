import os
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset


class ForestDataset(Dataset):

    def __init__(self, cfg, label_csv):

        self.main_dir = main_dir
        self.topo_image_dir = cfg.topo_dir
        self.labels = pd.read_csv(label_csv)

    def __getitem__(self, idx):

        row = self.labels.iloc[idx]

        image = np.load(
            os.path.join(
                self.topo_image_dir,
                row["filename"]
            )
        )

        image = torch.tensor(
            image,
            dtype=torch.float32
        ).unsqueeze(0)

        extra_features = torch.tensor([
            row["MAP"],
            row["rainfall_seasonality"],
            row["soil_type"]
        ], dtype=torch.float32)

        target = torch.tensor(
            row["forest_probability"],
            dtype=torch.float32
        )

        return image, extra_features, target

    def __len__(self):
        return len(self.labels)