import torch
from tqdm import tqdm
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

class Evaluator:
    def __init__(self, model, scaler, device):
        """
        Initialize the Evaluator class.

        Args:
            model: The PyTorch model to evaluate.
            scaler: The scaler used to normalize and inverse-transform the data.
            device: The device (CPU/GPU) to use for computations.
        """
        self.model = model
        self.scaler = scaler
        self.device = device

    def val_epoch(self, val_loader):
        """
        Perform validation and collect predictions and targets.

        Args:
            val_loader: DataLoader for validation data.

        Returns:
            tuple: test_predictions and test_targets as lists.
        """
        self.model.eval()
        test_predictions = []
        test_targets = []

        with torch.no_grad():
            for batch_X, batch_y in tqdm(val_loader, desc="Evaluation"):
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)

                # Forward pass
                outputs = self.model(batch_X, teacher_forcing_targets=batch_y)

                # Collect predictions and targets
                test_predictions.append(outputs.cpu().numpy())
                test_targets.append(batch_y[:, :, 0].squeeze().cpu().numpy())

        return test_predictions, test_targets

    def evaluate_predictions_per_timestep(self, test_predictions, test_targets):
        """
        Evaluate predictions per timestep.

        Args:
            test_predictions: List of predictions from val_epoch.
            test_targets: List of targets from val_epoch.

        Returns:
            dict: MAE, RMSE, and MAPE metrics per timestep.
        """
        # Concatenate predictions and targets
        predictions = np.concatenate(test_predictions, axis=0)
        targets = np.concatenate(test_targets, axis=0)

        # Rescale predictions and targets to their original range
        predictions_rescaled = self.scaler.inverse_transform(predictions)
        targets_rescaled = self.scaler.inverse_transform(targets)

        # Initialize metrics
        mae_per_timestep = []
        rmse_per_timestep = []
        mape_per_timestep = []

        # Calculate metrics for each timestep
        for t in range(predictions_rescaled.shape[1]):
            pred_t = predictions_rescaled[:, t]
            target_t = targets_rescaled[:, t]

            mae = mean_absolute_error(target_t, pred_t)
            rmse = np.sqrt(mean_squared_error(target_t, pred_t))
            mape = np.mean(np.abs((target_t - pred_t) / target_t)) * 100  # Avoid division by zero issues

            mae_per_timestep.append(mae)
            rmse_per_timestep.append(rmse)
            mape_per_timestep.append(mape)

        # Combine results in a dictionary
        results = {
            "MAE_per_timestep": mae_per_timestep,
            "RMSE_per_timestep": rmse_per_timestep,
            "MAPE_per_timestep": mape_per_timestep,
        }

        return results
