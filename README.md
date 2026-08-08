# Motor Pump Predictive System

This project is a small predictive maintenance workflow for a pump-like machine. In plain English, it does two main jobs:

1. It learns to recognize healthy versus faulty vibration patterns from the CWRU bearing dataset.
2. It prepares the project for later manual-answering workflows, model monitoring, and retraining when the incoming data changes.

The whole system is built in a way that is easy to understand: first it prepares the data, then it trains a few models, then it checks whether the data has drifted, and finally it can retrain when needed.

---

## What the project is trying to do

Think of this project as a tiny AI assistant for machine health.

- The prediction part listens to vibration signals and tries to decide whether the machine looks healthy or faulty.
- The monitoring part watches for changes in the data so the model does not silently become less trustworthy over time.
- The retraining part gives you a simple way to rebuild the model if the incoming data starts looking different from the training data.

---

## Folder-by-folder explanation

### data/
This is where the project keeps its raw and processed files.

- data/raw/: original downloaded data, such as the CWRU bearing recordings.
- data/processed/: cleaned, windowed, and feature-engineered data that the training scripts use.

You can think of this as the project’s pantry. The raw ingredients are kept separate from the prepared ingredients.

### models/
This folder is where trained model files are stored.

- The project saves trained classifiers here so you do not need to retrain them from scratch every time.
- The main saved model is meant to be reused by later scripts or API calls.

### notebooks/
This folder holds the interactive notebook used for exploration.

- The notebook is useful for trying ideas quickly.
- It now calls into the reusable scripts in the src folder instead of keeping all logic inline.

### reports/
This folder stores analysis outputs such as charts, notes, and drift-monitoring reports.

- If the data changes over time, the drift report helps you spot that change visually.
- This is where you can also keep notes about feature selection and model choices.

### src/
This is the heart of the project. It contains the code that does the real work.

#### src/features/
This folder contains the data-processing and model-training code.

- extract.py: computes the vibration features from each signal window.
- pipeline.py: creates windows from raw vibration signals and builds the feature table.
- run_pipeline.py: runs the data preparation workflow end to end.
- notebook_helpers.py: small helpers that make the notebook use the same pipeline as the scripts.
- model_training.py: trains and compares multiple models with cross-validation and evaluation.
- train_model.py: older simple wrapper for training a single model, kept for compatibility.

#### src/monitoring/
This folder contains the monitoring and retraining logic.

- drift.py: creates an Evidently drift report to see whether new data differs from the training data.
- retrain.py: checks whether retraining should happen and triggers the training flow if needed.

#### src/main.py
This is the FastAPI service entry point.

- It exposes a small API that can be used by another app or interface.
- It is meant to route user questions or requests to the right part of the system.

#### src/app.py
This is the Streamlit chat-style UI.

- It gives you a simple web interface where you can ask questions or interact with the system.
- It is helpful for demonstrating the workflow without building a larger app.

---

## Main scripts and what they do

### 1. Data preparation
Run this when you want to build the processed dataset from the raw vibration files:

```bash
uv run python -m src.features.run_pipeline
```

What it does:
- reads the CWRU vibration files,
- cuts them into overlapping windows,
- removes invalid or flat chunks,
- extracts features,
- saves the processed files to data/processed/.

### 2. Model training and comparison
Run this when you want to train and compare several models:

```bash
uv run python -m src.features.model_training
```

What it does:
- loads the processed feature table,
- splits the data into training and testing sets,
- trains several models,
- evaluates each model with cross-validation and test metrics,
- selects the best one for later use.

### 3. Drift monitoring
Run this when you want to compare new incoming data against the training data:

```bash
uv run python -m src.monitoring.drift
```

What it does:
- loads the reference feature set and a current feature set,
- creates a drift report in reports/,
- helps you decide whether the incoming data is changing enough to require retraining.

### 4. Retraining check
Run this when you want the project to decide whether a new model should be trained:

```bash
uv run python -m src.monitoring.retrain
```

What it does:
- compares the current feature table with the reference feature table,
- triggers the drift check,
- retrains the model if the data appears different enough to matter.

### 5. API server
Run this to start the FastAPI service:

```bash
uv run uvicorn src.main:app --reload
```

What it does:
- launches a local web API,
- allows another app or UI to send requests to the system.

### 6. Streamlit app
Run this to start the chat-style demo app:

```bash
uv run streamlit run src/app.py
```

What it does:
- opens a simple web page,
- lets you interact with the project through a basic interface.

---

## Why multiple models are used

The project now supports several model families instead of relying on just one.

- Random Forest is a strong baseline and is often a good first choice for tabular vibration features.
- Gradient Boosting can capture more complex patterns.
- SVM can work well when the data is well-scaled and the feature space is informative.

The system evaluates each one and keeps the best-performing option for later use.

---

## Why hyperparameter tuning matters

A model can perform very differently depending on its settings.

For example:
- a Random Forest with too few trees may underfit,
- too many trees may waste time,
- an SVM with poor settings may struggle on noisy data.

The current workflow uses a simple, practical evaluation approach with cross-validation so you can compare models and see which one actually generalizes well.

---

## Why monitoring and retraining matter

Machine data often changes over time.

This can happen when:
- the machine ages,
- the operating environment changes,
- the sensor setup changes,
- the vibration pattern becomes noisier or different.

When that happens, a model trained on older data may start making worse predictions. Drift monitoring helps catch that, and retraining gives you a path to update the model.

---

## Recommended workflow

1. Run the pipeline to create the processed feature data.
2. Train and compare the available models.
3. Review the metrics and select the best model.
4. Use the drift report to monitor new data.
5. Retrain when the data changes enough to make the old model less reliable.

---

## Reproducible runtime with Docker

This repository includes containerized runtime for both services:

- FastAPI backend on port `8000`
- Streamlit UI on port `8501`

### Prerequisites

- Docker Desktop (or Docker Engine)
- OpenAI API key

### Run with Docker Compose

1. Set your OpenAI API key in your shell:

```powershell
$env:OPENAI_API_KEY="your_key_here"
```

2. Build and start both containers:

```powershell
docker compose up --build -d
```

3. Open the UI:

- `http://localhost:8501`

4. Stop containers:

```powershell
docker compose down
```

### Optional: ingest manuals from inside Docker

If you need to rebuild vector chunks:

```powershell
docker compose run --rm api uv run python -m src.rag.ingest
```

---

## Reproducible infrastructure with Terraform (Docker provider)

Terraform in this project provisions the same two Docker containers and network.

Location:

- `infra/terraform/`

### Prerequisites

- Terraform `>= 1.6`
- Docker running locally

### Run with Terraform

1. Go to the Terraform directory:

```powershell
cd infra/terraform
```

2. Create a tfvars file from the example:

```powershell
copy terraform.tfvars.example terraform.tfvars
```

3. Edit `terraform.tfvars` and set `openai_api_key`.

4. Initialize and apply:

```powershell
terraform init
terraform apply
```

5. Destroy when done:

```powershell
terraform destroy
```

Terraform outputs:

- `api_url` (default `http://localhost:8000`)
- `ui_url` (default `http://localhost:8501`)

---

## Notes

This project is a practical starter system rather than a fully production-grade monitoring platform. It is designed to be understandable, modular, and easy to extend.

If you later want to make it more advanced, the next natural steps would be:
- add real hyperparameter tuning grids,
- save training history and metrics to CSV or JSON,
- connect the API to live sensor data,
- add automated retraining rules and alerts.
