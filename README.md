# Motor Pump Diagnostic Assistant

The Motor Pump Diagnostic Assistant is an end-to-end AI application for detecting bearing faults, answering maintenance questions, and monitoring diagnostic reliability over time. It combines vibration-signal analysis with a grounded equipment-manual assistant so a maintenance user can move from **"What is wrong with this pump?"** to **"What should I do next?"** in one interface.

The project uses the public CWRU bearing dataset as its diagnostic training source and a pump equipment manual as its knowledge base. The application can:

1. Extract vibration features from raw `.npz` files and classify healthy or faulty bearing behavior.
2. Retrieve relevant manual passages and generate answers with citations to the source document.
3. Expose the diagnostic and RAG workflows through a FastAPI service and Streamlit interface.
4. Replay holdout sensor rows as a live feed, trigger repeated-fault alerts, collect user feedback, and display monitoring metrics.
5. Evaluate model accuracy, retrieval strategies, and answer-prompt styles before selecting the active approach.

This is a practical diagnostic assistant rather than an autonomous repair system. Its predictions support maintenance decisions; they do not replace inspection, safety procedures, or qualified engineering judgment.

---

## Problem we are solving

Motor pumps can develop bearing and mechanical faults before a complete breakdown occurs. In many maintenance settings, the warning signs are buried in vibration recordings, while the instructions needed to inspect, remove, replace, or safely maintain a component are spread across technical manuals. This creates three practical problems:

- **Faults are difficult to identify early:** a technician may need to inspect large vibration files manually or rely on experience to recognize abnormal patterns.
- **Diagnosis and maintenance guidance are disconnected:** a model may flag an abnormal reading without explaining what the manual recommends doing next.
- **Diagnostic quality can degrade over time:** changes in sensors, operating conditions, or machine wear can make an older model less reliable without an obvious warning.

The Motor Pump Diagnostic Assistant addresses this by combining vibration-based fault classification with retrieval-augmented manual guidance. It returns a predicted condition and confidence, grounds maintenance answers in source passages, monitors data and feedback, and provides alerts or retraining signals when the diagnostic workflow needs attention.

The intended users are maintenance technicians, reliability engineers, and developers evaluating a reproducible predictive-maintenance workflow. The project focuses on decision support: it helps a person investigate a pump condition and find relevant procedures, while leaving final safety and repair decisions to qualified personnel.

---

## Diagnostic assistant workflow

The system has two connected paths:

- **Vibration diagnosis:** raw pump-like vibration data is windowed, transformed into statistical and spectral features, scored by the selected classifier, and returned with a predicted condition and confidence.
- **Maintenance guidance:** a user asks a question in plain language, the assistant retrieves relevant manual chunks from Chroma, reranks them with hybrid retrieval, and asks the LLM to answer only from the supplied evidence.

Monitoring connects the two paths. The app can replay known sensor rows, identify repeated high-confidence fault predictions, record user feedback, compare current data with the reference distribution, and provide a retraining path when drift is detected.

---

## How to run

### Prerequisites

- Python `3.12` or newer
- [uv](https://docs.astral.sh/uv/)
- An OpenAI API key for embeddings and manual-question answering
- The CWRU bearing files under `data/raw/CWRU_Bearing_NumPy/`
- Docker Desktop, only if using the containerized setup

### Local setup

From the repository root, install the pinned project environment:

```powershell
uv sync
```

Create a `.env` file in the repository root and add your OpenAI key:

```env
OPENAI_API_KEY=your_key_here
```

Build the processed vibration features and train the diagnostic model:

```powershell
uv run python -m src.features.run_pipeline
uv run python -m src.features.model_training
```

Ingest the equipment manual into Chroma when the manual index needs to be created or refreshed:

```powershell
uv run python -m src.rag.ingest
```

Start the API in one terminal:

```powershell
uv run uvicorn src.main:app --reload --port 8000
```

Start the Streamlit assistant in a second terminal:

```powershell
uv run streamlit run src/app.py --server.port 8501
```

Open the application at [http://localhost:8501](http://localhost:8501). The API is available at [http://localhost:8000](http://localhost:8000), with interactive endpoint documentation at [http://localhost:8000/docs](http://localhost:8000/docs).

### Docker Compose setup

With Docker Desktop running, set the API key in PowerShell and start both services:

```powershell
$env:OPENAI_API_KEY="your_key_here"
docker compose up --build -d
```

Use the same URLs:

- Streamlit assistant: [http://localhost:8501](http://localhost:8501)
- FastAPI service: [http://localhost:8000](http://localhost:8000)

Stop the containers when finished:

```powershell
docker compose down
```

To rebuild the manual index inside the API container:

```powershell
docker compose run --rm api uv run python -m src.rag.ingest
```

### Evaluation and monitoring commands

Generate the evaluation reports and drift report with:

```powershell
uv run python scripts/generate_accuracy_report.py
uv run python scripts/evaluate_retrieval.py
uv run python scripts/evaluate_llm_prompts.py
uv run python -m src.monitoring.drift
```

The Streamlit **Monitoring Dashboard** displays collected feedback. Feedback is stored locally in `data/feedback/feedback.jsonl` and is created after a user submits a rating in the interface.

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

### src/main.py
This is the FastAPI service entry point for the diagnostic assistant.

- It exposes manual-question, sensor-prediction, raw-file upload, replay, and feedback endpoints.
- It routes maintenance questions to the RAG workflow and diagnostic questions to the vibration classifier.
- It returns source information, confidence values, and alert results for the UI or other clients.

### src/app.py
This is the Streamlit interface for the Motor Pump Diagnostic Assistant.

- It supports manual chat, raw vibration uploads, manual feature prediction, and live monitoring replay.
- It collects feedback on answers and predictions.
- Its monitoring dashboard shows feedback, source quality, confidence, and activity trends.

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

## Retrieval evaluation

The RAG flow is evaluated with two retrieval approaches:

- vector search from Chroma
- hybrid search that combines vector retrieval with keyword reranking

The best-performing option is documented in:

- reports/retrieval_evaluation_report.md
- reports/retrieval_evaluation_report.json

Run it with:

```bash
uv run python scripts/evaluate_retrieval.py
```

## LLM prompt evaluation

The final-answer prompt is evaluated with two styles:

- short answers
- detailed answers

The selected style is documented in:

- reports/llm_evaluation_report.md
- reports/llm_evaluation_report.json

Run it with:

```bash
uv run python scripts/evaluate_llm_prompts.py
```

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

## User feedback and monitoring dashboard

The app now collects user feedback on manual answers, sensor predictions, and live-monitoring events. Feedback is stored in:

- data/feedback/feedback.jsonl

The Streamlit monitoring dashboard shows:

- feedback counts by rating
- feedback counts by route
- source-quality mix
- average confidence by rating
- feedback over time
- recent feedback entries

---

## Recommended diagnostic workflow

1. Run the ingestion or feature pipeline to create the diagnostic data and manual knowledge base.
2. Train and compare the available fault-classification models.
3. Review the model, retrieval, and LLM evaluation reports.
4. Start the API and Streamlit assistant.
5. Upload a vibration file or enter extracted features to receive a diagnostic result.
6. Ask the manual assistant for maintenance guidance and review its cited sources.
7. Use live replay, drift monitoring, feedback, and retraining checks to maintain diagnostic reliability.

---

## Model accuracy (documented)

The latest evaluated classifier metrics are documented in:

- reports/model_accuracy_report.md
- reports/model_accuracy_report.json

Evaluation setup:
- Data source: data/processed/features.parquet
- Features: rms, kurtosis, crest_factor, dominant_freq, spectral_energy, spectral_entropy
- Holdout split: 20% test set, stratified by label, random_state=42
- Holdout size: 613 rows

Latest holdout result:
- Overall accuracy: 1.0000
- Macro F1: 1.0000
- Weighted F1: 1.0000

To regenerate this report:

```bash
uv run python scripts/generate_accuracy_report.py
```

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
