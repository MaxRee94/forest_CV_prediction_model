import torch.nn as nn
from torch.utils.data import random_split
from torch.utils.data import DataLoader
import torch
import PT_dataset
import PT_model
import PT_traintest
from types import SimpleNamespace


def main(cfg):
    # Split dataset into training and testing sets
    dataset = PT_dataset.ForestDataset(
        image_dir=cfg.topo_dir,
        label_csv=cfg.topo_dir
    )

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size

    train_dataset, test_dataset = random_split(
        dataset,
        [train_size, test_size]
    )


    # Create dataloaders to feed the data into the neural network
    cfg.train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True
    )

    cfg.test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.batch_size,
        shuffle=False
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    model = PT_model.ForestCNN().to(device)


    # Define the optimizer and criterion
    cfg.criterion = nn.MSELoss()

    cfg.optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-3
    )


    # Train the model
    PT_traintest.train(cfg)

    # Evaluate the model
    PT_traintest.evaluate(cfg)


if __name__ == "__main__":
    cfg = SimpleNamespace()
    cfg.batch_size = 32
    cfg.num_epochs = 10
    cfg.device = "cuda" if torch.cuda.is_available() else "cpu"

    main(cfg)


