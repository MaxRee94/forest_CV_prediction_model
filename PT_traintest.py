

def do_training(cfg):

    for epoch in range(cfg.num_epochs):

        cfg.model.train()

        running_loss = 0.0

        for images, features, targets in cfg.train_loader:

            images = images.to(cfg.device)
            features = features.to(cfg.device)
            targets = targets.to(cfg.device)

            cfg.optimizer.zero_grad()

            predictions = cfg.model(
                images,
                features
            )

            loss = cfg.criterion(
                predictions,
                targets
            )

            loss.backward()

            cfg.optimizer.step()

        avg_loss = running_loss / len(cfg.train_loader)

        print(
            f"Epoch {epoch+1}/{cfg.num_epochs}"
            f"  Loss = {avg_loss:.5f}"
        )



def evaluate(cfg):
    cfg.model.eval()

    test_loss = 0

    with cfg.torch.no_grad():

        for images, targets in cfg.test_loader:

            images = images.to(cfg.device)
            targets = targets.to(cfg.device)

            predictions = cfg.model(images)

            loss = cfg.criterion(
                predictions,
                targets
            )

            test_loss += loss.item()

    test_loss /= len(cfg.test_loader)

    print(
        f"Test loss: {test_loss:.5f}"
    )



