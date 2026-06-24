import torch.nn as nn
import torch


class ForestCNN(nn.Module):

    def __init__(self):

        super().__init__()

        # Image branch
        self.cnn = nn.Sequential(

            nn.Conv2d(1,16,3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16,32,3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32,64,3,padding=1),
            nn.ReLU(),

            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten()
        )

        # Climate/soil branch
        self.tabular = nn.Sequential(

            nn.Linear(3,16),
            nn.ReLU(),

            nn.Linear(16,16),
            nn.ReLU()
        )

        # Combined branch
        self.head = nn.Sequential(

            nn.Linear(64 + 16, 32),
            nn.ReLU(),

            nn.Linear(32,1),
            nn.Sigmoid()
        )

    def forward(self, image, features):

        image_features = self.cnn(image)

        tabular_features = self.tabular(features)

        combined = torch.cat(
            [image_features,
             tabular_features],
            dim=1
        )

        return self.head(combined).squeeze(1)