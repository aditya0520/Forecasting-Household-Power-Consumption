import yaml
import torch
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader
from prefect import flow, task, get_run_logger
import mlflow
from mlflow.pytorch import log_model
from data_ingestion import load_and_clean_data
from feature_engineering import create_features
from dataset import TimeSeriesDataset
from training_utils import Trainer
from evaluation import Evaluator
from loss import combined_loss
from model import Model


@task
def load_config(config_path):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config

# Data ingestion task
@task
def ingest_data(file_name):
    raw_data = load_and_clean_data(file_name)
    return raw_data

# Feature engineering task
@task
def engineer_features(raw_data):
    processed_data = create_features(raw_data)
    return processed_data

# Data splitting task
@task
def split_data(processed_data, train_split):
    train_size = int(len(processed_data) * train_split)
    train_data = processed_data[:train_size]
    test_data = processed_data[train_size:]
    return train_data, test_data

# Dataset and DataLoader preparation task
@task
def prepare_dataloader(train_data, test_data, seq_length, prediction_horizon, batch_size, num_workers):
    scaler = MinMaxScaler()
    train_dataset = TimeSeriesDataset(train_data, seq_length, prediction_horizon, scaler=scaler)
    test_dataset = TimeSeriesDataset(test_data, seq_length, prediction_horizon, scaler=scaler)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, scaler

# Model initialization task
@task
def initialize_model(input_size, hidden_size, num_layers, fcc_intermediate, prediction_horizon, device):
    model = Model(input_size, hidden_size, num_layers, fcc_intermediate, prediction_horizon).to(device)
    return model

# Training task
@task
def train_model(model, train_loader, val_loader, optimizer, scheduler, device, num_epochs, combined_loss, config):
    logger = get_run_logger()
    trainer = Trainer(model, optimizer, combined_loss, device)
    train_losses, val_losses = [], []

    # Start MLflow run
    mlflow.start_run()

    # Log configuration parameters to MLflow
    mlflow.log_params(config)

    for epoch in range(num_epochs):
        train_loss = trainer.train_epoch(train_loader)
        val_loss = trainer.val_epoch(val_loader)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        # Log metrics to MLflow
        mlflow.log_metrics({"train_loss": train_loss, "val_loss": val_loss}, step=epoch)

        logger.info(
            f"Epoch [{epoch + 1}/{num_epochs}] - "
            f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
            f"Learning Rate: {float(optimizer.param_groups[0]['lr']):.6f}"
        )
        
        scheduler.step(val_loss)

    # Log the model to MLflow
    log_model(model, "model")

    mlflow.end_run()

    return train_losses, val_losses

# Evaluation task
@task
def evaluate_model(model, val_loader, scaler, device):
    evaluator = Evaluator(model, scaler, device)
    test_predictions, test_targets = evaluator.val_epoch(val_loader)
    results = evaluator.evaluate_predictions_per_timestep(test_predictions, test_targets)
    return results

# Prefect flow
@flow
def training_pipeline(config_path):
    # Step 1: Load Config
    config = load_config(config_path)

    # Step 2: Data Ingestion
    raw_data = ingest_data(config["file_name"])

    # Step 3: Feature Engineering
    processed_data = engineer_features(raw_data)

    # Step 4: Split Data
    train_data, test_data = split_data(processed_data, config["train_split"])

    # Step 5: Prepare Dataloader
    train_loader, val_loader, scaler = prepare_dataloader(
        train_data,
        test_data,
        config["seq_length"],
        config["prediction_horizon"],
        config["batch_size"],
        config["num_workers"],
    )

    # Step 6: Initialize Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = initialize_model(
        config["input_size"],
        config["hidden_size"],
        config["num_layers"],
        config["fcc_intermediate"],
        config["prediction_horizon"],
        device,
    )

    # Step 7: Optimizer and Scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    scheduler_params = config["scheduler"]
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode=scheduler_params["mode"],
        factor=scheduler_params["factor"],
        patience=scheduler_params["patience"],
    )

    # # Step 8: Train Model
    # train_losses, val_losses = train_model(
    #     model, train_loader, val_loader, optimizer, scheduler, device, config["num_epochs"], combined_loss, config
    # )

    model.load_state_dict(torch.load("../miscellaneous/lstm_model.pth", map_location=torch.device('cpu')))

    # input_example = np.random.rand(config["batch_size"], config["seq_length"], config["input_size"])
    mlflow.start_run()
    log_model(model, "model")

    # Step 9: Evaluate Model
    # results = evaluate_model(model, val_loader, scaler, device)

    # Step 10: Save Results
    # print("Training and Evaluation Complete")
    # print("Results:", results)

    # mlflow.log_metric("MAE_mean", np.mean(results["MAE_per_timestep"]))
    # mlflow.log_metric("RMSE_mean", np.mean(results["RMSE_per_timestep"]))
    # mlflow.log_metric("MAPE_mean", np.mean(results["MAPE_per_timestep"]))
    mlflow.end_run()

# Run the pipeline
if __name__ == "__main__":
    training_pipeline("../config/config.yaml")
