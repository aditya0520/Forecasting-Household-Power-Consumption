# Forecasting Household Power Consumption

This project demonstrates a robust pipeline for forecasting household power consumption using a deep learning model deployed with MLflow. The solution integrates MLOps technologies for tracking, automation, and deployment.

---

## Objective

The objective of this project is to predict the power utilization of a household over the next 10 minutes. This was achieved by training a deep learning model using an LSTM encoder-decoder architecture with teacher-forcing.

---

## Features

- **Time Series Forecasting**: Utilizes deep learning models for accurate predictions using features like sine and cosine of the day of the week.
- **Scaled Input**: The input `Global_active_power` variable is scaled using MinMaxScaler from scikit-learn.
- **Automated Pipeline**: Orchestrated with Prefect for seamless data processing and training.
- **Experiment Tracking**: MLflow for logging metrics, hyperparameters, and managing model versions.
- **Model Deployment**: Serve models using MLflow for real-time inference.
- **Batch Predictions**: Supports real-time and batch inference with REST API.

---

## Dataset

The dataset used for this project is available at: [Individual household electric power consumption dataset](https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption).

- **Description**: Measurements of electric power consumption in one household with a one-minute sampling rate over a period of almost 4 years.

---

## Technologies Used

- **Python**: Core programming language.
- **PyTorch**: Framework for building and training deep learning models.
- **MLflow**:
  - Experiment Tracking: Logs hyperparameters, metrics, and models.
  - Model Registry: Manages versioned models.
  - Model Serving: Serves trained models via REST API.
- **Prefect**: Automates data ingestion, feature engineering, training, and evaluation.
- **scikit-learn**: Preprocessing and feature scaling.

---

## Project Structure

```
├── data/
│   ├── household_power_consumption.txt       # Training dataset
├── config/
│   ├── config.yaml                           # Configuration file for parameters
├── scripts/
│   ├── data_ingestion.py                     # Load and clean raw data
│   ├── feature_engineering.py                # Generate features
│   ├── dataset.py                            # TimeSeriesDataset class
│   ├── model.py                              # PyTorch model (Encoder-Decoder)
│   ├── training_utils.py                     # Training and validation utilities
│   ├── evaluation.py                         # Evaluation metrics and functions
│   ├── loss.py                               # Combined loss functions
│   ├── train.py                              # Prefect pipeline for training
│   ├── inference.py                          # Utilising the deployed model for prediction

├── requirements.txt                          # Python dependencies
└── README.md                                 # Project documentation
```

---

## Setup and Installation

### Clone the Repository
```bash
git clone https://github.com/yourusername/forecasting-household-power-consumption.git
cd forecasting-household-power-consumption
```

### Set Up a Virtual Environment
```bash
python3 -m venv myenv
source myenv/bin/activate  # On Windows, use `myenv\Scripts\activate`
pip install -r requirements.txt
```

### Set Up MLflow
Start the MLflow tracking UI:
```bash
mlflow ui
```
Access the UI at [http://127.0.0.1:5000](http://127.0.0.1:5000).

### Setup Prefect
Start the Prefect UI:
```bash
prefect server start
```

### Run the Prefect Pipeline
Execute the training pipeline:
```bash
python scripts/train.py
```

### Serve the Model with MLflow
Deploy the trained model:
```bash
mlflow models serve -m runs:/<run_id>/model -p 1234
```

### Serve the Model Using MLflow
Load the trained model and use MLflow for inference:
```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

---

## Usage

### Train the Model
Configure training parameters in `config/config.yaml` and execute:
```bash
python scripts/train.py
```

### Utilize the Model
Run the following script:
```bash
python scripts/inference.py
```
---

## Key MLOps Features

### Prefect Automation
- Automates the pipeline from data ingestion to model training and evaluation.
- Provides robust monitoring and retry mechanisms.

### MLflow Experiment Tracking
- Logs hyperparameters, metrics, and models.
- Tracks model versions and provides a central model registry.

### Deployment
- Locally deployed using MLflow's model serving.

---

## Future Enhancements

- **Model Monitoring**: Integrate tools like Evidently AI for drift detection.
- **Cloud Deployment**: Deploy the pipeline on cloud services like AWS, Azure, or GCP.
- **CI/CD Integration**: Automate the pipeline with GitHub Actions or Jenkins.

---

