import yaml
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, DataLoader
from data_ingestion import load_and_clean_data
from feature_engineering import create_features
from dataset import TimeSeriesDataset
from training_utils import Trainer
from evaluation import Evaluator
from loss import combined_loss
from model import Model
import torch

# Load config
with open("../config/config.yaml", "r") as file:
    config = yaml.safe_load(file)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load and process data
file_name = config["file_name"]
raw_data = load_and_clean_data(file_name)
processed_data = create_features(raw_data)

# Train-test split
train_split = config["train_split"]
train_size = int(len(processed_data) * train_split)
train_data = processed_data[:train_size]
test_data = processed_data[train_size:]

# Dataset and DataLoader
seq_length = config["seq_length"]
prediction_horizon = config["prediction_horizon"]

scaler = MinMaxScaler()
train_dataset = TimeSeriesDataset(train_data, seq_length, prediction_horizon, scaler=scaler)
test_dataset = TimeSeriesDataset(test_data, seq_length, prediction_horizon, scaler=scaler)

batch_size = config["batch_size"]
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=config["num_workers"])
val_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=config["num_workers"])

# Model
input_size = config["input_size"]
hidden_size = config["hidden_size"]
num_layers = config["num_layers"]
fcc_intermediate = config["fcc_intermediate"]
model = Model(input_size, hidden_size, num_layers, fcc_intermediate, prediction_horizon).to(device)

# Optimizer and Scheduler
learning_rate = config["learning_rate"]
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
scheduler_params = config["scheduler"]
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode=scheduler_params["mode"],
    factor=scheduler_params["factor"],
    patience=scheduler_params["patience"]
)

# Training
num_epochs = config["num_epochs"]
train_losses = []
val_losses = []

trainer = Trainer(model, optimizer, combined_loss, device)

for epoch in range(num_epochs):
    train_loss = trainer.train_epoch(train_loader)
    train_losses.append(train_loss)

    val_loss, test_predictions, test_targets = trainer.val_epoch(val_loader)
    val_losses.append(val_loss)

    print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Curr LR: {float(optimizer.param_groups[0]['lr'])}")

    scheduler.step(val_loss)

# Evaluation
evaluator = Evaluator(model, scaler, device)
test_predictions, test_targets = evaluator.val_epoch(val_loader)
results = evaluator.evaluate_predictions_per_timestep(test_predictions, test_targets)

print("Evaluation Results:")
print("MAE per timestep:", results["MAE_per_timestep"])
print("RMSE per timestep:", results["RMSE_per_timestep"])
print("MAPE per timestep:", results["MAPE_per_timestep"])
