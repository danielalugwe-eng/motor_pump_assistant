# Folder and Script Guide

This document explains the repository structure in plain language and describes what each script is doing.

## Top-level folders

### data/
Purpose: stores raw and processed data.

Why it matters:
- Raw data should stay untouched.
- Processed data can be transformed into windows and features without reloading the original files.

Subfolders:
- raw/: original vibration recordings and downloaded datasets
- processed/: cleaned windows, feature tables, and intermediate outputs

### models/
Purpose: stores trained model files.

Why it matters:
- You do not need to retrain the model every time you want to test or score new data.
- The saved model is the trained decision engine.

### notebooks/
Purpose: contains interactive analysis notebooks.

Why it matters:
- Notebooks are useful for exploration and experiments.
- They should call the same reusable logic as the scripts whenever possible.

### reports/
Purpose: stores charts, notes, and monitoring output.

Why it matters:
- This is where you preserve evidence of model evaluation and data drift.
- It helps you compare results over time.

### src/
Purpose: the main source code folder.

Why it matters:
- This is where the real pipeline lives.
- It contains data processing, feature engineering, training, monitoring, and API logic.

## Core source subfolders

### src/features/
This is the foundation of the predictive workflow.

### src/monitoring/
This handles drift detection and retraining logic.

### src/rag/
This is the manual-answering part of the project. It uses embeddings and a vector database to search a PDF manual.

## Script-by-script explanation

### root-level main.py
File: main.py

What it does:
- Defines a simple entry point that prints a short message.

Why it exists:
- It serves as a placeholder or basic starter script.
- It is not the main prediction engine.

### src/data_utils.py
What it does:
- Loads the latest sensor CSV if available.
- Builds a simple risk score from the mean RMS and kurtosis values.

Key code pieces:
- load_latest_sensor_data(): reads a CSV file from data/processed/latest_week.csv
- build_breakdown_signal(): calculates a risk score and assigns a simple status

Why it exists:
- It gives you a simple way to turn incoming sensor features into a basic health signal.
- In this project it acts as a lightweight readiness layer for future real-time use.

### src/features/extract.py
What it does:
- Converts a single vibration window into a small set of numerical features.

Key code pieces:
- extract_features(window): computes mean, std, rms, skew, kurtosis, ptp, crest_factor, dominant_freq, spectral_energy, and spectral_entropy
- The FFT part uses rfft and rfftfreq to capture the frequency content of the signal

Why it exists:
- Raw vibration windows are too detailed for a simple classifier.
- Features summarize the signal into a compact form that the model can learn from.

### src/features/pipeline.py
What it does:
- Loads raw vibration files
- Uses the filename to infer the label
- Splits each long signal into overlapping windows
- Keeps only valid windows
- Saves the windows as a DataFrame
- Builds the feature table from those windows

Key code pieces:
- get_label(): maps filenames to labels such as Normal, InnerRace, Ball, or OuterRace
- window_signal(): creates overlapping windows from the original vibration signal
- build_windows(): reads the dataset files and saves windows to data/processed/windows.pkl
- build_feature_table(): turns windows into features and saves data/processed/features.parquet

Why it exists:
- This is the core data preparation step.
- Without it, the training script would not have a clean and consistent dataset.

### src/features/run_pipeline.py
What it does:
- Runs the whole data preparation workflow in one step.

Key code pieces:
- build_windows()
- build_feature_table()

Why it exists:
- It makes the pipeline easy to run from a single command.
- This is the command you use when you want fresh processed data.

### src/features/model_training.py
What it does:
- Loads the feature table
- Splits the data into training and testing sets
- Trains several model families
- Evaluates them with cross-validation and classification reports
- Saves the best model to models/best_fault_classifier.pkl

Key code pieces:
- build_models(): defines Random Forest, Gradient Boosting, and SVM pipelines
- evaluate_models(): trains and evaluates the models
- train_best_model(): selects the best-performing model and saves it

Why it exists:
- This is the main training and evaluation engine.
- It helps you compare different algorithms rather than relying on a single model choice.

### src/features/train_model.py
What it does:
- Provides a lighter wrapper around the model training process.

Why it exists:
- It is a simpler interface for older or smaller workflows.
- It is useful when you want a quick training entry point without touching the full training module.

### src/features/notebook_helpers.py
What it does:
- Holds helper utilities that make notebook-based exploration consistent with the scripts.

Why it exists:
- It keeps notebook code shorter and reduces duplication.
- It improves consistency between interactive experiments and production-style scripts.

### src/main.py
What it does:
- Starts a FastAPI web service.
- Routes questions to either the predictive side or the manual-search side.

Key code pieces:
- Query model: defines the expected request structure
- ask(): checks whether the user is asking a prediction-style question or a manual question
- It uses load_latest_sensor_data() and build_breakdown_signal() for prediction-style requests
- It uses search_manual() for manual questions

Why it exists:
- It exposes the system as an API service.
- It is useful if you want to integrate the project into another app or a web interface.

### src/app.py
What it does:
- Provides a small Streamlit user interface.

Why it exists:
- It makes the project easier to demonstrate.
- You can type a question into a web app and see the system respond.

### src/rag/ingest.py
What it does:
- Reads a PDF manual
- Splits the text into chunks
- Sends the chunks to OpenAI embeddings
- Stores the embeddings in a Chroma database

Key code pieces:
- ingest_manual(): reads the PDF and loads it into the collection
- RecursiveCharacterTextSplitter(): breaks long text into manageable chunks

Why it exists:
- This enables the manual-answering side of the project.
- It is the knowledge base for the RAG portion of the system.

### src/rag/search.py
What it does:
- Converts the user’s question into an embedding
- Searches the Chroma database for the closest matching text chunks

Why it exists:
- This is the retrieval step for the RAG workflow.
- It lets the system pull relevant manual content before answering.

### src/monitoring/drift.py
What it does:
- Builds an Evidently drift report.
- Compares a reference dataset with a current dataset.

Why it exists:
- Machine behavior changes over time.
- Drift monitoring helps you detect when new data is becoming meaningfully different from the training set.

### src/monitoring/retrain.py
What it does:
- Checks whether retraining should happen.
- Triggers drift detection and model retraining if the data changed.

Why it exists:
- A model that was good yesterday can become less reliable later.
- This gives you an automated path to refresh the model.

### tests/test_feature_pipeline.py
What it does:
- Tests that feature extraction returns the expected set of feature keys.

Why it exists:
- It ensures the core feature extraction function remains stable.
- It prevents simple regressions when editing the code.
