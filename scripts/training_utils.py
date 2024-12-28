import torch
from tqdm import tqdm

class Trainer:
    def __init__(self, model, optimizer, combined_loss, device):
        """
        Initialize the Trainer class.

        Args:
            model: The PyTorch model to train and evaluate.
            optimizer: The optimizer used for training.
            combined_loss: The loss function to optimize.
            device: The device (CPU/GPU) to use for computations.
        """
        self.model = model
        self.optimizer = optimizer
        self.combined_loss = combined_loss
        self.device = device

    def train_epoch(self, train_loader):
        """
        Perform one epoch of training.

        Args:
            train_loader: DataLoader for training data.

        Returns:
            float: Average training loss for the epoch.
        """
        self.model.train()
        train_loss = 0.0

        for batch_X, batch_y in tqdm(train_loader, desc="Training"):
            batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)

            # Forward pass
            outputs = self.model(batch_X, teacher_forcing_targets=batch_y)
            loss = self.combined_loss(outputs, batch_y[:, :, 0].squeeze())

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()

        return train_loss / len(train_loader)

    def val_epoch(self, val_loader):
        """
        Perform one epoch of validation.

        Args:
            val_loader: DataLoader for validation data.

        Returns:
            tuple: Average validation loss, predictions, and targets.
        """
        self.model.eval()
        val_loss = 0.0
        test_predictions = []
        test_targets = []

        with torch.no_grad():
            for batch_X, batch_y in tqdm(val_loader, desc="Validation"):
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)

                # Forward pass
                outputs = self.model(batch_X, teacher_forcing_targets=batch_y)
                loss = self.combined_loss(outputs, batch_y[:, :, 0].squeeze())

                val_loss += loss.item()

                # Collect predictions and targets for evaluation
                test_predictions.append(outputs.cpu().numpy())
                test_targets.append(batch_y[:, :, 0].squeeze().cpu().numpy())

        return val_loss / len(val_loader), test_predictions, test_targets
