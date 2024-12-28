import sys
import torch
from tqdm import tqdm
from prefect import get_run_logger


class Trainer:
    def __init__(self, model, optimizer, combined_loss, device):
        self.model = model
        self.optimizer = optimizer
        self.combined_loss = combined_loss
        self.device = device

    def train_epoch(self, train_loader):
        self.model.train()
        train_loss = 0.0
        logger = get_run_logger()

        # Add `file=sys.stdout` to ensure `tqdm` works in Prefect or silent environments
        for batch_idx, (batch_X, batch_y) in enumerate(tqdm(train_loader, desc="Training", file=sys.stdout)):
            batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)

            # Forward pass
            outputs = self.model(batch_X, teacher_forcing_targets=batch_y)
            loss = self.combined_loss(outputs, batch_y[:, :, 0].squeeze())

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()
            if batch_idx % 100 == 0:
                logger.info(f"Batch {batch_idx}/{len(train_loader)} - Loss: {loss.item():.4f}")

        return train_loss / len(train_loader)

    def val_epoch(self, val_loader):
        self.model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for batch_X, batch_y in tqdm(val_loader, desc="Validation", file=sys.stdout):
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)

                # Forward pass
                outputs = self.model(batch_X, teacher_forcing_targets=batch_y)
                loss = self.combined_loss(outputs, batch_y[:, :, 0].squeeze())

                val_loss += loss.item()

        return val_loss / len(val_loader)
