# Motor Pump Predictive System — Detailed Explanation Like You Are 5

This file explains the whole project from the very beginning to the very end.

The goal is simple:
- explain what the project does
- explain what each big file does
- explain the imports
- explain the functions
- explain how data moves through the system
- explain the RAG + LLM part
- explain the prediction part
- explain it like you are learning for the first time

## 1. The big picture

Imagine you built **two smart helpers**.

Helper 1 is a **machine doctor**.
- It listens to vibration data.
- It tries to decide if the machine looks healthy or faulty.

Helper 2 is a **manual reader**.
- It reads the machine manual.
- It answers questions by looking inside that manual.

Then you put both helpers inside one app.
- If you ask a manual question, the manual helper answers.
- If you give vibration data, the machine doctor answers.

That is this project.

## 2. The full journey from beginning to end

This project works in this order:

1. Get raw vibration data from the CWRU dataset.
2. Cut long signals into smaller windows.
3. Turn each window into smart summary numbers called features.
4. Train models to classify healthy vs faulty behavior.
5. Save the best model.
6. Read the PDF manual and split it into text chunks.
7. Store those chunks in a vector database.
8. When the user asks a question, search the manual for the best chunks.
9. Give those chunks to the LLM.
10. Let the LLM answer using only that manual evidence.
11. Show everything in a Streamlit app.

## 3. Folder map

### `data/`
This is the pantry.
- `data/raw/` = original files
- `data/processed/` = cleaned and transformed files
- `data/top_ex.pdf` = the manual PDF used by the chat assistant
- `data/chroma_db/` = stored vector search database for the manual

### `models/`
This is where trained machine-learning models are saved.

### `src/`
This is the real engine room of the project.

### `md/`
This is where explanation documents live.

### `tests/`
This is where tests live.

### `scripts/`
This is where helper scripts for quick checks and debugging live.

## 4. The libraries and what they mean

### `numpy`
Used for fast math on vibration signals.
Think of it like a super calculator for big lists of numbers.

### `pandas`
Used for tables.
Think of it like Excel inside Python.

### `scipy`
Used for signal math and statistics.
It helps calculate things like frequency and kurtosis.

### `scikit-learn`
Used for machine learning.
It trains the model that predicts healthy vs faulty conditions.

### `joblib`
Used for saving and loading trained models.

### `fastapi`
Used to build the backend API.
It creates endpoints like `/ask`, `/predict`, and `/predict_raw`.

### `uvicorn`
Used to run the FastAPI server.

### `streamlit`
Used to create the browser app.

### `openai`
Used to create embeddings and ask the LLM for final answers.

### `chromadb`
Used to store manual text chunks in a searchable way.
It is the memory shelf for the manual.

### `pypdf`
Used to read text from PDF files.

### `langchain-text-splitters`
Used to cut long manual text into smaller chunks.

### `python-dotenv`
Used to load secret settings like the OpenAI API key from a `.env` file.

### `python-multipart`
Used so FastAPI can accept uploaded files like `.npz` vibration files.

### `requests`
Used by the Streamlit app to talk to the backend API.

### `evidently`
Used to detect data drift.
That means it checks whether new data is starting to look different from the old data.

## 5. The feature extraction file

File: `src/features/extract.py`

This file takes one vibration window and turns it into useful features.

### Imports

`from __future__ import annotations`
- This helps Python understand type hints more safely.
- You can think of it as a future-friendly typing helper.

`from typing import Any`
- `Any` means a variable can be many kinds of things.
- In this file it is not heavily used, but it supports flexible typing.

`import numpy as np`
- Brings in `numpy`.
- `np` is just a short nickname.

`from scipy.fft import rfft, rfftfreq`
- `rfft` turns a signal into frequency space.
- `rfftfreq` tells you what frequencies belong to the FFT bins.

`from scipy.stats import kurtosis, skew`
- `kurtosis` measures how spiky a signal is.
- `skew` measures whether the signal leans more one way than the other.

### Constant

`FS = 12000`
- This is the sampling rate.
- It means the signal was measured 12,000 times per second.

### Function: `extract_features(window)`
This is one of the most important functions in the whole project.

It takes one vibration window and computes:
- `mean`
- `std`
- `rms`
- `skew`
- `kurtosis`
- `ptp`
- `crest_factor`
- `dominant_freq`
- `spectral_energy`
- `spectral_entropy`

#### Step by step

`window = np.asarray(window, dtype=float).reshape(-1)`
- Make sure the input becomes a clean 1D list of numbers.

`if window.size == 0:`
- If there are no numbers, stop with an error.

`feats = {}`
- Create an empty box to store features.

`feats["mean"] = float(np.mean(window))`
- Average value of the signal.

`feats["std"] = float(np.std(window))`
- Standard deviation.
- How spread out the numbers are.

`feats["rms"] = float(np.sqrt(np.mean(window**2)))`
- Root mean square.
- A popular vibration strength measure.

`feats["skew"] = float(skew(window))`
- Shows whether the signal shape leans left or right.

`feats["kurtosis"] = float(kurtosis(window))`
- Shows whether the signal has strong spikes.
- Very useful for fault detection.

`feats["ptp"] = float(np.ptp(window))`
- Peak-to-peak distance.
- Highest minus lowest point.

`feats["crest_factor"] = ...`
- Tells you how sharp the highest spike is compared to the overall signal strength.

`freqs = rfftfreq(len(window), 1 / FS)`
- Computes the real FFT frequencies.

`fft_vals = np.abs(rfft(window))`
- Converts the signal to frequency space and takes magnitude.

`dominant_freq`
- The loudest frequency.

`spectral_energy`
- Total amount of frequency energy.

`spectral_entropy`
- Measures how messy or spread-out the frequencies are.

At the end, the function returns the dictionary of features.

## 6. The feature pipeline file

File: `src/features/pipeline.py`

This file turns raw `.npz` files into windows, then windows into features.

### Imports

`from pathlib import Path`
- Makes file paths easier and safer to use.

`from typing import Any`
- Allows flexible dictionaries and rows.

`import numpy as np`
- Needed for loading and processing the signal.

`import pandas as pd`
- Needed for tables.

`from .extract import extract_features`
- Imports the feature extractor from the same folder.

### Constants

`FS = 12000`
- Sample rate.

`WINDOW_SIZE = 2048`
- Each signal chunk has 2048 points.

`OVERLAP = 0.5`
- Windows overlap by 50%.

`STEP = int(WINDOW_SIZE * (1 - OVERLAP))`
- This decides how far the next window moves.
- With 50% overlap, the next window starts halfway through the last one.

### Function: `get_label(filename)`
This reads the file name and decides the class.

Examples:
- `Normal` -> healthy
- `_IR_` -> inner race fault
- `_B_` -> ball fault
- `OR@6` -> outer race fault

It returns a label string or `None`.

### Function: `window_signal(signal, window_size, step)`
This takes one long signal and cuts it into many smaller windows.

Why?
- Models learn better from many short examples than from one giant signal.

### Function: `build_windows(data_dir)`
This is the raw-data preparation step.

What it does:
1. Finds `.npz` files.
2. Loads the `DE` signal from each file.
3. Skips dead or flat signals.
4. Cuts each signal into windows.
5. Skips flat windows.
6. Stores rows with:
   - `window`
   - `label`
   - `source_file`
7. Saves them to `data/processed/windows.pkl`.

### Function: `build_feature_table(input_path)`
This loads the windows and converts each one into features.

What it does:
1. Load `windows.pkl`.
2. Run `extract_features(window)` for each row.
3. Build a DataFrame of features.
4. Add the label column.
5. Save to `data/processed/features.parquet`.

This file is the bridge from raw signals to machine-learning data.

## 7. The model training file

File: `src/features/model_training.py`

This trains and compares multiple machine-learning models.

### Imports

`joblib`
- Saves trained models.

`pandas`
- Loads feature tables.

`RandomForestClassifier`, `GradientBoostingClassifier`, `SVC`
- Three different machine-learning models.

`classification_report`
- Builds evaluation metrics.

`StratifiedKFold`, `cross_val_score`, `train_test_split`
- Used to evaluate models properly.

`Pipeline`, `StandardScaler`
- Used to scale features before SVM.

### Constant

`SELECTED_FEATURES = [...]`
- These are the six important features used for training.

### Function: `build_models()`
Returns a dictionary of models:
- random forest
- gradient boosting
- svm pipeline

### Function: `evaluate_models(features_path)`
What it does:
1. Load the feature table.
2. Split into `X` and `y`.
3. Split training and test sets.
4. Train each model.
5. Evaluate each one.
6. Return their scores.

### Function: `train_best_model(features_path, model_path)`
What it does:
1. Evaluate all models.
2. Pick the one with highest cross-validation accuracy.
3. Rebuild that model.
4. Fit it again.
5. Save it to `models/best_fault_classifier.pkl`.

### Function: `main()`
This is just a runner.
It trains the best model and prints the result.

## 8. The older training wrapper

File: `src/features/train_model.py`

This is a simpler older training script.
It only trains one model: a Random Forest.

It exists so the project has a simpler compatibility path.

Main parts:
- load features
- split train/test
- run cross-validation
- fit model
- save model to `models/mixer_fault_classifier_v1.pkl`
- return metrics

## 9. The pipeline runner

File: `src/features/run_pipeline.py`

This is a very small file.
Its job is just to run the full feature pipeline.

### Function: `main()`
It does:
- `build_windows()`
- `build_feature_table()`
- print success message

This is helpful because it lets one command run the whole preparation flow.

## 10. The sensor utility file

File: `src/data_utils.py`

This file helps the prediction side of the app.

### Imports

`BytesIO`
- Lets Python treat bytes like a file.
- Very useful for uploaded `.npz` files.

`Path`
- Safe file paths.

`Any`
- Flexible typing.

`numpy`, `pandas`
- Signal and table tools.

`extract_features`
- Converts raw windows into features.

`WINDOW_SIZE`, `window_signal`
- Reuses the same window logic as the training pipeline.

### Function: `load_latest_sensor_data(path)`
Loads a CSV file with sensor features.
If the file does not exist, it raises an error.

### Function: `build_breakdown_signal(features)`
Creates a simple fallback risk score:
- 40% RMS
- 60% Kurtosis

If score > 1.0, status becomes `high_risk`.
Otherwise `normal`.

This is not the fancy trained model.
It is a simple backup rule.

### Function: `load_cwru_npz_signal(npz_bytes, signal_key)`
This loads raw vibration arrays from an uploaded `.npz` file.

What it does:
1. Open bytes like a file.
2. Check which arrays exist.
3. Prefer keys like `DE`, `FE`, `BA`.
4. Choose the requested key if available.
5. Convert the selected array into a clean 1D float signal.

### Function: `extract_vibration_features_from_npz(npz_bytes, signal_key)`
This is important for the app.

What it does:
1. Load the raw signal.
2. Window it if it is long enough.
3. Skip flat windows.
4. Extract features for each window.
5. Average them.
6. Return a row with:
   - the averaged features
   - chosen signal key
   - number of windows
   - raw signal length

This is how uploaded raw data becomes prediction-ready.

## 11. The RAG ingest file

File: `src/rag/ingest.py`

This file reads the PDF manual and stores it in the vector database.

### Imports

`os`
- Used to read environment variables like the API key.

`Path`
- Safer path handling.

`chromadb`
- Vector database.

`load_dotenv`
- Loads `.env` values.

`RecursiveCharacterTextSplitter`
- Splits long manual text into chunks.

`OpenAI`
- Generates embeddings.

`PdfReader`
- Reads text from PDF pages.

### `load_dotenv()`
This loads the `.env` file so `OPENAI_API_KEY` becomes available.

### Function: `ingest_manual(pdf_path, collection_name, skip_embeddings)`
This is the heart of the manual indexing process.

What it does:
1. Choose a PDF file or a folder of PDFs.
2. Make sure the PDF exists.
3. If it is a folder, sort all PDFs.
4. Create a text splitter.
5. Create or recreate the Chroma collection.
6. Read each PDF page.
7. Extract text.
8. Split page text into chunks.
9. Save metadata for each chunk:
   - source file name
   - page number
   - chunk number
   - file order
10. If no text exists, raise an error.
11. If embeddings are skipped or API key is missing, store text only.
12. Otherwise create embeddings in batches of 10.
13. Save embeddings, text, metadata, and ids into ChromaDB.

### Why batching matters
If you embed too much at once, requests can become slow or fail.
Small batches are safer.

### Why `skip_embeddings` matters
Sometimes OpenAI may not be available.
The project can still store text-only chunks as a fallback.

## 12. The RAG search file

File: `src/rag/search.py`

This file searches the indexed manual.

### Function: `search_manual(query, k, collection_name)`
What it does:
1. Open the Chroma collection.
2. Read the OpenAI API key.
3. Turn the user question into an embedding.
4. Ask Chroma for the nearest `k` chunks.
5. Return documents and metadata.

This is like asking the manual:
- “Which few paragraphs are closest to this question?”

## 13. The RAG LLM file

File: `src/rag/llm.py`

This file decides where a question goes and how the prompt is built.

### Keyword sets

`PREDICTIVE_KEYWORDS`
- words that smell like prediction questions
- example: `predict`, `fault`, `vibration`, `risk`

`MANUAL_KEYWORDS`
- words that smell like manual/procedure questions
- example: `replace`, `electrical`, `fuse`, `maintenance`, `modification`

### Function: `route_query(text)`
This counts keyword matches and decides:
- `predictor`
- or `rag`

### Function: `build_rag_context(results)`
This takes the retrieved chunks and arranges them nicely.

What it does:
1. Read documents and metadata.
2. Pair each chunk with its page/source.
3. Sort by source, file order, page, and chunk.
4. Add headers like `[top_ex.pdf page 39]`
5. Join everything into one text block.

### Function: `build_rag_prompt(query, context, answer_style)`
This builds the exact instruction for the LLM.

It tells the LLM:
- use only the manual excerpts
- answer directly if evidence exists
- cite sources
- keep it short or detailed depending on the style
- ask a follow-up only if necessary

### Function: `extract_answer_text(response)`
LLM responses can have different shapes.
This function safely extracts the answer text.

## 14. The API file

File: `src/main.py`

This is the brain that connects everything together.

### Imports

`BytesIO`, `os`, `Any`, `Literal`
- utility helpers for files, environment, flexible types, and allowed string values

`numpy`, `joblib`
- math and model loading

`FastAPI`, `File`, `Form`, `UploadFile`
- build endpoints and accept uploaded files

`OpenAI`
- used for the LLM call

`BaseModel`
- validates incoming request data

`src.data_utils`, `src.rag.llm`, `src.rag.search`
- connect prediction and manual systems

### Global setup

`app = FastAPI(...)`
- creates the API application

`client = OpenAI(...)`
- creates OpenAI client

`clf = joblib.load(...)`
- loads the trained prediction model if it exists

`SELECTED_FEATURES = [...]`
- defines the exact feature order used for prediction

### Helper: `_build_sources(results)`
Turns retrieval metadata into human-readable source strings.
Example:
- `top_ex.pdf page 39`

### Helper: `_source_quality_from_count(source_count)`
Turns number of sources into:
- `high`
- `medium`
- `low`
- `none`

### Class: `Query`
Defines request body for `/ask`.
Fields:
- `text`
- `answer_style`

### Class: `SensorPayload`
Defines request body for `/predict`.
Fields:
- `rms`
- `kurtosis`
- `crest_factor`
- `dominant_freq`
- `spectral_energy`
- `spectral_entropy`

### Function: `predict_from_feature_row(feature_row)`
This uses the trained model if possible.

What it does:
1. Build feature list in the correct order.
2. If model exists:
   - predict class probabilities
   - choose best class
   - return label and confidence
3. If model is missing or fails:
   - use fallback risk rule
   - return normal or high_risk

### Endpoint: `root()`
Simple health endpoint.
Returns a message showing the API is alive.

### Endpoint: `ask(q)`
This is the manual/chat endpoint.

What it does:
1. Read user text.
2. Route question to predictor or RAG.
3. If predictor:
   - load latest sensor CSV
   - build simple risk answer
4. If RAG:
   - search manual chunks
   - build context
   - build prompt
   - ask LLM
   - return answer, sources, source quality, preview
5. If LLM fails:
   - return retrieval-only evidence fallback
6. If retrieval fails:
   - return a polite user-friendly fallback

This endpoint is the main chat brain.

### Endpoint: `predict(sensor)`
Takes already-extracted feature values and predicts condition.

### Endpoint: `predict_raw(file, signal_key)`
This is the raw upload endpoint.

What it does:
1. Check that uploaded file is `.npz`.
2. Read file bytes.
3. Extract features from raw signal.
4. Run prediction.
5. Return prediction plus metadata.

## 15. The Streamlit app file

File: `src/app.py`

This file creates the web app you use in the browser.

### Imports

`streamlit as st`
- Streamlit UI library

`requests`
- used to call the backend API

### Page setup

`st.set_page_config(...)`
- page title and icon

`st.title(...)`
- main heading

`st.write(...)`
- helpful intro text

### Session state

This app uses `st.session_state` to remember things between reruns.

Stored values:
- `history` = previous chat messages
- `manual_mode` = old manual mode placeholder
- `query` = current query box value
- `pending_query` = value that will prefill next follow-up

### Mode switch

`mode = st.radio(...)`
Lets the user pick:
- Manual Chat
- Sensor Prediction

### Manual Chat branch

Features:
- answer style selector
- question form
- send button
- clear history button
- ask a follow-up button
- source-quality badge
- context preview expander

How it works:
1. User types question.
2. App sends POST request to `/ask`.
3. API returns answer and sources.
4. App stores response in chat history.
5. App shows answer with source quality.

### Sensor Prediction branch

Two input modes:
- Upload raw CWRU `.npz`
- Enter extracted features manually

#### Raw upload mode
1. User uploads `.npz` file.
2. App sends file to `/predict_raw`.
3. API extracts features.
4. API predicts condition.
5. App shows:
   - answer
   - file metadata
   - extracted features
   - healthy/bad interpretation
   - confidence badge

#### Manual feature mode
1. User types six feature values.
2. App sends them to `/predict`.
3. API predicts condition.
4. App shows result and confidence badge.

## 16. Monitoring files

### File: `src/monitoring/drift.py`
This checks whether data has drifted.

Function: `run_drift_check(reference, current, output_path)`
What it does:
1. Load reference features
2. Load current features
3. Build Evidently report
4. Save HTML report

### File: `src/monitoring/retrain.py`
This decides whether retraining is needed.

Function: `retrain_if_needed(reference_path, current_path)`
What it does:
1. Load old and new feature tables
2. If they are identical, skip retraining
3. If different:
   - run drift check
   - train best model again
   - return retraining info

## 17. Package file

File: `pyproject.toml`

This file tells Python:
- the project name
- Python version needed
- all required packages

Important dependencies:
- `fastapi`, `uvicorn` for backend
- `streamlit` for frontend
- `openai`, `chromadb`, `pypdf`, `langchain-text-splitters` for RAG
- `numpy`, `pandas`, `scipy`, `scikit-learn`, `joblib` for prediction
- `evidently` for drift
- `requests` for UI API calls
- `python-multipart` for file upload support

## 18. How the manual assistant works, like you are 5

Imagine you ask:
- “Where is the fuse located?”

The system does this:
1. It sees this is a manual question.
2. It searches the PDF chunks.
3. It finds the best pages.
4. It builds a small evidence pack.
5. It gives that evidence pack to the LLM.
6. The LLM answers using only those pages.
7. The answer comes back with sources.

That is RAG.

## 19. How the predictor works, like you are 5

Imagine you give the app a raw vibration file.

The system does this:
1. Open the file.
2. Pick the signal inside.
3. Cut the signal into windows.
4. Turn each window into features.
5. Average the features.
6. Feed them to the model.
7. Get a label like `Normal`, `Ball`, `InnerRace`, or `OuterRace`.
8. Show whether this looks healthy or bad.

## 20. Healthy vs bad readings

### Healthy
Usually means:
- prediction is `Normal`
- confidence is good
- RMS and Kurtosis are not extreme
- signal is smoother

### Bad
Usually means:
- prediction is `Ball`, `InnerRace`, or `OuterRace`
- confidence may be high for a fault class
- RMS is larger
- Kurtosis is larger
- crest factor is larger
- signal is more spiky

## 21. Commands from beginning to end

### Step 1: Build processed data
```powershell
uv run python -m src.features.run_pipeline
```

### Step 2: Train best model
```powershell
uv run python -m src.features.model_training
```

### Step 3: Ingest manual into vector DB
```powershell
uv run python -m src.rag.ingest
```

### Step 4: Start backend
```powershell
uv run uvicorn src.main:app --reload --host 127.0.0.1 --port 8000
```

### Step 5: Start UI
```powershell
uv run streamlit run src/app.py
```

## 22. What to learn first if you are brand new

Learn in this order:
1. `src/features/extract.py`
2. `src/features/pipeline.py`
3. `src/features/model_training.py`
4. `src/data_utils.py`
5. `src/rag/ingest.py`
6. `src/rag/search.py`
7. `src/rag/llm.py`
8. `src/main.py`
9. `src/app.py`

Why this order?
Because it follows the life of the data:
- raw signal
- windows
- features
- model
- manual chunks
- search
- prompt
- API
- app

## 23. Final simple summary

This whole project is really just this:

- One part reads machine vibrations and predicts faults.
- One part reads a manual and answers questions.
- One backend joins them together.
- One app lets people use both.

So if you remember one sentence, remember this:

**This project teaches a computer to read machine vibrations and machine manuals, then explain what it finds in a simple app.**

## 24. Tiny files and helper files (do not skip these)

This section explains the smaller files too.

### File: `main.py` (project root)
This is a tiny starter file.

Code idea:
- it defines `main()`
- `main()` prints a hello message
- if the file is run directly, it calls `main()`

Why it exists:
- mostly as a simple starter entry point from `uv init`
- it is not the real API app

### File: `src/features/__init__.py`
This file contains only a short docstring.

What that means:
- it tells Python this folder is a package
- it gives a one-line description of the feature-engineering area

### File: `src/features/notebook_helpers.py`
This file helps notebooks reuse the same real pipeline code.

Imports:
- `Path` for safe paths
- `pandas as pd` for data tables
- `build_feature_table` and `build_windows` from `pipeline.py`

Functions:

`prepare_data()`
- runs `build_windows()`
- then runs `build_feature_table()`
- returns the feature DataFrame

`project_root()`
- returns the project root folder path
- useful inside notebooks when paths get confusing

### File: `src/test_llm.py`
This is a quick test file for checking whether OpenAI chat calls work.

Imports:
- `os` to read environment variables
- `load_dotenv` to load `.env`
- `OpenAI` to create the API client

Function: `test_llm(prompt=...)`
- loads the API key
- creates OpenAI client
- sends a tiny prompt like “Say hello”
- returns the LLM text

Bottom block:
- if run directly, print the LLM response
- if it fails, print the exception

Why it exists:
- a quick diagnostics tool for “is my API key and chat model working?”

## 25. All scripts inside `scripts/`

These are the workshop tools. They are not the main product. They help test, debug, or inspect the system.

### File: `scripts/check_index.py`
Purpose:
- check whether the Chroma collection exists and how many items it has

Imports:
- `sys` and `Path` to add project root to import path
- `chromadb` to talk to the vector DB

Behavior:
- open `manual_chunks`
- try `collection.count()`
- if that fails, try a fallback query
- print the count or print an error

### File: `scripts/check_openai.py`
Purpose:
- inspect the installed OpenAI package version and available class attributes

Behavior:
- writes debug info to `terminal_openai_debug.txt`
- checks if `OpenAI` has `chat`, `responses`, `embeddings`, and `completions`

Why useful:
- helps when API SDK versions behave differently

### File: `scripts/check_rag_query.py`
Purpose:
- run one RAG search query and save the raw result

Behavior:
- imports `search_manual`
- asks “Where are the fuses located?”
- writes results to `scripts/rag_query_result.json`

### File: `scripts/chroma_debug.py`
Purpose:
- inspect Chroma behavior, especially collection creation and document adding

Behavior:
- prints Chroma version
- checks client methods
- creates a test collection
- tries adding a document without explicit embeddings

Why useful:
- this helped debug the no-embedding fallback path

### File: `scripts/debug_ask.py`
Purpose:
- test the whole ask-flow from search to final answer

Behavior:
- prints working directory and API key presence
- runs `route_query()`
- runs `search_manual()`
- builds context
- directly calls `ask()`

Why useful:
- good for tracing where a failure happens in the manual-answer path

### File: `scripts/debug_llm_flow.py`
Purpose:
- test the exact RAG + LLM pipeline and save verbose details

Behavior:
- runs a manual search
- builds context and prompt
- creates OpenAI client
- sends the prompt to the LLM
- saves everything into `terminal_llm_debug.txt`

Why useful:
- a deep debugging script for prompt and response problems

### File: `scripts/probe_rag.py`
Purpose:
- do a small RAG probe with a simple query

Behavior:
- searches for `maintenance`
- prints number of documents found
- prints first chunk of context

### File: `scripts/run_ingest_all.py`
Purpose:
- run manual ingestion from a script

Behavior:
- adds project root to `sys.path`
- imports `ingest_manual`
- calls `ingest_manual('data/manuals')`
- prints chunk count

### File: `scripts/run_ingest_capture.py`
Purpose:
- run ingestion and save the result to a text file

Behavior:
- calls `ingest_manual('data')`
- writes either `ingested_count=...` or `error=...` to `scripts/ingest_result.txt`

### File: `scripts/smoke_test.py`
Purpose:
- tiny sanity check for core helper functions

Behavior:
- checks `route_query()`
- checks `build_rag_context()`
- checks that `extract_features()` returns the expected keys

Why useful:
- fast “did we break the basics?” script

### File: `scripts/test_api_flow.py`
Purpose:
- tiny direct call to `ask()` without HTTP

Behavior:
- creates a fake query object `Q`
- prints `ask(Q(...))`

### File: `scripts/test_ask.py`
Purpose:
- send one live HTTP request to `/ask`

Behavior:
- asks “Where are the fuses located?”
- saves JSON result to `scripts/last_ask_response.json`
- or saves raw text to `scripts/last_ask_response.txt`

### File: `scripts/test_e2e_endpoints.py`
Purpose:
- end-to-end test using FastAPI `TestClient`

Behavior:
- tests `/predict_raw` with a real CWRU `.npz` file
- tests `/ask` with a manual question
- writes the results to `scripts/e2e_test_result.json`

### File: `scripts/test_e2e_console.py`
Purpose:
- similar to the previous file, but prints to the console instead of writing JSON

Behavior:
- tests `/predict_raw`
- tests `/ask` with a predictor-style question
- prints status and response previews

### File: `scripts/test_question.py`
Purpose:
- test one specific broad manual question

Behavior:
- asks “can I perform an electronic modification on this equipment”
- saves result to `scripts/last_question_response.json`

### File: `scripts/test_modification_question.py`
Purpose:
- almost the same as `test_question.py`
- created specifically for the electronic-modification case

Behavior:
- saves result to `scripts/modification_question_response.json`

### File: `scripts/manual_random_questions_test.py`
Purpose:
- run many random manual questions and collect pass/fail summary

Behavior:
- shuffles a list of manual questions
- calls `/ask` for each
- marks failures if status is bad or answer contains error text
- writes summary to `scripts/manual_random_questions_results.json`

### File: `scripts/manual_random_questions_console.py`
Purpose:
- same idea as above, but prints directly to console instead of writing JSON

Behavior:
- prints question, status, route, source quality, and short answer preview
- counts total failures

## 26. The test files in `tests/`

These are formal tests used by `pytest`.

### File: `tests/test_rag_llm.py`
Purpose:
- verify the RAG helper behavior

Test 1: `test_route_query_detects_predictive_requests()`
- checks that a prediction-style question routes to `predictor`
- checks that a maintenance question routes to `rag`

Test 2: `test_build_rag_context_joins_document_chunks()`
- checks that simple chunks join correctly when no metadata is present

### File: `tests/test_feature_pipeline.py`
Purpose:
- verify feature extraction basics

Test: `test_extract_features_returns_expected_keys()`
- creates a sine wave
- runs `extract_features()`
- checks that all expected keys exist
- checks important values are finite

## 27. Every import pattern explained simply

You asked for imports too, so here is the simple rule:

### Standard library imports
Examples:
- `os`
- `sys`
- `json`
- `random`
- `pathlib.Path`
- `io.BytesIO`

These come with Python itself.

### Third-party imports
Examples:
- `numpy`
- `pandas`
- `streamlit`
- `fastapi`
- `openai`
- `chromadb`

These are installed from packages listed in `pyproject.toml`.

### Local project imports
Examples:
- `from src.rag.search import search_manual`
- `from src.features.extract import extract_features`

These connect one file in your project to another file in your project.

## 28. Final “nothing skipped” checklist

The guide now explicitly covers:

- `main.py`
- `src/app.py`
- `src/main.py`
- `src/data_utils.py`
- `src/test_llm.py`
- `src/features/__init__.py`
- `src/features/extract.py`
- `src/features/pipeline.py`
- `src/features/run_pipeline.py`
- `src/features/notebook_helpers.py`
- `src/features/model_training.py`
- `src/features/train_model.py`
- `src/rag/ingest.py`
- `src/rag/search.py`
- `src/rag/llm.py`
- `src/monitoring/drift.py`
- `src/monitoring/retrain.py`
- all helper files in `scripts/`
- both test files in `tests/`
- `pyproject.toml`
- `README.md`

So now the explanation does not skip any script in the repo.

## 29. Line-by-line explanation blocks for every script

This appendix goes through every Python script again in a more literal, line-by-line style.

Important note:
- To keep this readable, I group nearby lines when they are doing one tiny job together.
- Blank lines are not explained one by one because they only separate sections.
- Every real code line is still covered in meaning.

---

### File: `main.py`

- `def main():`
   - Create a simple function named `main`.
- `print("Hello from motor-pump-predictive-system!")`
   - Print a tiny hello message.
- `if __name__ == "__main__":`
   - Check whether this file is being run directly.
- `main()`
   - Run the `main` function.

### File: `src/features/__init__.py`

- `"""Feature engineering helpers for the predictive maintenance workflow."""`
   - This is a docstring.
   - It describes what this package is for.

### File: `src/features/notebook_helpers.py`

- `from __future__ import annotations`
   - Future-friendly typing behavior.
- `from pathlib import Path`
   - Import safe file path handling.
- `import pandas as pd`
   - Import pandas with nickname `pd`.
- `from .pipeline import build_feature_table, build_windows`
   - Import two local helper functions from `pipeline.py`.
- `def prepare_data() -> pd.DataFrame:`
   - Define a helper that prepares data and returns a DataFrame.
- `build_windows()`
   - Create windowed vibration data from raw files.
- `return build_feature_table()`
   - Build the feature table and return it.
- `def project_root() -> Path:`
   - Define a helper that returns the project root path.
- `return Path(__file__).resolve().parents[1]`
   - Use the current file location, go upward, and return the root folder.

### File: `src/test_llm.py`

- `from __future__ import annotations`
   - Future typing behavior.
- `import os`
   - Needed to read environment variables.
- `from dotenv import load_dotenv`
   - Load `.env` file values.
- `from openai import OpenAI`
   - Import the OpenAI client.
- `def test_llm(prompt: str = "Say hello in one short sentence.") -> str:`
   - Define a function that checks whether the LLM can answer.
- `load_dotenv()`
   - Load `.env` so the API key becomes available.
- `api_key = os.getenv("OPENAI_API_KEY")`
   - Read the OpenAI key.
- `if not api_key:`
   - Stop if the key is missing.
- `raise RuntimeError(...)`
   - Raise a helpful error.
- `client = OpenAI(api_key=api_key)`
   - Create the OpenAI client.
- `response = client.chat.completions.create(...)`
   - Send a chat request to the model.
- `return response.choices[0].message.content or ""`
   - Return the model text safely.
- `except Exception as exc:`
   - Catch failures.
- `raise RuntimeError(...) from exc`
   - Wrap the original failure in a cleaner message.
- `if __name__ == "__main__":`
   - If this file is run directly...
- `print(test_llm())`
   - Print the LLM result.
- `except Exception as exc: print(exc)`
   - Print any error if the test fails.

### File: `src/features/extract.py`

- `from __future__ import annotations`
   - Future typing behavior.
- `from typing import Any`
   - Import flexible type helper.
- `import numpy as np`
   - Import NumPy.
- `from scipy.fft import rfft, rfftfreq`
   - Import FFT helpers.
- `from scipy.stats import kurtosis, skew`
   - Import statistical helpers.
- `FS = 12000`
   - Set sample rate constant.
- `def extract_features(window: np.ndarray) -> dict[str, float]:`
   - Define feature extraction function.
- `window = np.asarray(window, dtype=float).reshape(-1)`
   - Force the window into a clean 1D float array.
- `if window.size == 0:`
   - Check for empty input.
- `raise ValueError("window must not be empty")`
   - Stop if there is no data.
- `feats: dict[str, float] = {}`
   - Create a dictionary to store features.
- `feats["mean"] = ...`
   - Store average value.
- `feats["std"] = ...`
   - Store standard deviation.
- `feats["rms"] = ...`
   - Store root mean square energy.
- `feats["skew"] = ...`
   - Store skewness.
- `feats["kurtosis"] = ...`
   - Store kurtosis.
- `feats["ptp"] = ...`
   - Store peak-to-peak range.
- `feats["crest_factor"] = ...`
   - Store crest factor.
- `freqs = rfftfreq(len(window), 1 / FS)`
   - Build frequency axis.
- `fft_vals = np.abs(rfft(window))`
   - Take FFT magnitude.
- `feats["dominant_freq"] = ...`
   - Store strongest frequency.
- `feats["spectral_energy"] = ...`
   - Store total energy in frequency space.
- `feats["spectral_entropy"] = ...`
   - Store how spread-out the spectrum is.
- `return feats`
   - Return the finished feature dictionary.

### File: `src/features/pipeline.py`

- `from __future__ import annotations`
   - Future typing behavior.
- `from pathlib import Path`
   - Safe file paths.
- `from typing import Any`
   - Flexible row typing.
- `import numpy as np`
   - Numerical operations.
- `import pandas as pd`
   - Data tables.
- `from .extract import extract_features`
   - Import feature extractor.
- `FS = 12000`, `WINDOW_SIZE = 2048`, `OVERLAP = 0.5`, `STEP = ...`
   - Define constants for signal windowing.
- `def get_label(filename: Path) -> str | None:`
   - Define label detector based on file name.
- `name = filename.stem`
   - Remove extension and keep file name body.
- `if "Normal" in name:`
   - Healthy class check.
- `if "_IR_" in name:`
   - Inner race fault check.
- `if "_B_" in name:`
   - Ball fault check.
- `if "OR@6" in name:`
   - Outer race fault check.
- `return None`
   - Unknown file pattern.
- `def window_signal(signal: np.ndarray, ...) -> list[np.ndarray]:`
   - Define function that slices signal into windows.
- `windows: list[np.ndarray] = []`
   - Start an empty window list.
- `for start in range(...):`
   - Move across the signal step by step.
- `windows.append(signal[start : start + window_size])`
   - Save each chunk.
- `return windows`
   - Return all windows.
- `def build_windows(data_dir: str | Path | None = None) -> pd.DataFrame:`
   - Define raw-to-window pipeline step.
- `data_dir = Path(...)`
   - Choose default raw data folder.
- `files = list(...glob(...)) + list(...glob(...))`
   - Find selected `.npz` files.
- `rows: list[dict[str, Any]] = []`
   - Create storage for output rows.
- `for file_path in files:`
   - Loop over every matching raw file.
- `label = get_label(file_path)`
   - Decide class label.
- `if label is None: continue`
   - Skip unknown file types.
- `with np.load(file_path) as data:`
   - Open `.npz` file safely.
- `signal = data["DE"]`
   - Use drive-end signal.
- `if np.std(signal) < 1e-6: continue`
   - Skip flat signals.
- `for chunk in window_signal(signal):`
   - Split the signal into windows.
- `if np.std(chunk) < 1e-6: continue`
   - Skip flat windows.
- `rows.append({...})`
   - Save window, label, and source file.
- `df = pd.DataFrame(rows)`
   - Turn rows into a table.
- `if not df.empty:`
   - Only save if data exists.
- `output_path = Path("data/processed/windows.pkl")`
   - Choose save location.
- `output_path.parent.mkdir(...)`
   - Ensure folder exists.
- `df.to_pickle(output_path)`
   - Save windows table.
- `return df`
   - Return the table.
- `def build_feature_table(input_path: str | Path | None = None) -> pd.DataFrame:`
   - Define window-to-feature pipeline step.
- `input_path = Path(...)`
   - Choose default window file.
- `df = pd.read_pickle(input_path)`
   - Load windows.
- `feature_rows = [extract_features(window) for window in df["window"]]`
   - Compute features for each window.
- `features_df = pd.DataFrame(feature_rows)`
   - Turn features into table.
- `features_df["label"] = df["label"].values`
   - Add class labels back.
- `output_path = Path("data/processed/features.parquet")`
   - Choose save path.
- `output_path.parent.mkdir(...)`
   - Ensure folder exists.
- `features_df.to_parquet(output_path, index=False)`
   - Save feature table.
- `return features_df`
   - Return it.

### File: `src/features/run_pipeline.py`

- `from __future__ import annotations`
   - Future typing behavior.
- `from pathlib import Path`
   - Path helper, though not deeply used here.
- `from .pipeline import build_feature_table, build_windows`
   - Import the two pipeline stages.
- `def main() -> None:`
   - Define runner function.
- `build_windows()`
   - Make windows from raw data.
- `build_feature_table()`
   - Make features from windows.
- `print("Pipeline completed successfully.")`
   - Print success message.
- `if __name__ == "__main__": main()`
   - Run it when called directly.

### File: `src/features/train_model.py`

- `from __future__ import annotations`
   - Future typing behavior.
- `from pathlib import Path`
   - Path helper.
- `import joblib`
   - Save trained model.
- `import pandas as pd`
   - Read feature table.
- `from sklearn...`
   - Import random forest, metrics, and split helpers.
- `SELECTED_FEATURES = [...]`
   - Define feature order.
- `def train_classifier(...) -> tuple[RandomForestClassifier, dict[str, object]]:`
   - Define simple random-forest training function.
- `features_path = Path(...)`, `model_path = Path(...)`
   - Resolve save and load paths.
- `df = pd.read_parquet(features_path)`
   - Load features.
- `X = df[SELECTED_FEATURES]`, `y = df["label"]`
   - Split features and labels.
- `train_test_split(...)`
   - Make train and test data.
- `clf = RandomForestClassifier(...)`
   - Create classifier.
- `cv_scores = cross_val_score(...)`
   - Evaluate model on training folds.
- `clf.fit(X_train, y_train)`
   - Train classifier.
- `preds = clf.predict(X_test)`
   - Predict on test set.
- `model_path.parent.mkdir(...)`
   - Ensure model folder exists.
- `joblib.dump(clf, model_path)`
   - Save model.
- `results = {...}`
   - Build result summary.
- `return clf, results`
   - Return model and metrics.
- `def main() -> None:`
   - Define CLI runner.
- `_, results = train_classifier()`
   - Train the model.
- `print(results["cv_accuracy"])`
   - Print CV score.
- `print(results["classification_report"])`
   - Print report.
- `if __name__ == "__main__": main()`
   - Run on direct execution.

### File: `src/features/model_training.py`

- `from __future__ import annotations`
   - Future typing behavior.
- `from pathlib import Path`, `from typing import Any`
   - Path and flexible typing helpers.
- `import joblib`, `import pandas as pd`
   - Save model and load tables.
- `from sklearn...`
   - Import all ML tools used for comparing models.
- `SELECTED_FEATURES = [...]`
   - Define the six chosen input features.
- `def build_models() -> dict[str, Any]:`
   - Return a dictionary of models.
- `"random_forest": RandomForestClassifier(...)`
   - Build random forest option.
- `"gradient_boosting": GradientBoostingClassifier(...)`
   - Build gradient boosting option.
- `"svm": Pipeline([...])`
   - Build scaled SVM option.
- `def evaluate_models(...) -> dict[str, Any]:`
   - Define evaluation function.
- `features_path = Path(...)`
   - Choose feature file.
- `df = pd.read_parquet(features_path)`
   - Load feature table.
- `X = df[SELECTED_FEATURES]`, `y = df["label"]`
   - Split inputs and labels.
- `train_test_split(...)`
   - Create train/test sets.
- `results: dict[str, Any] = {}`
   - Storage for model metrics.
- `for name, model in build_models().items():`
   - Loop over all models.
- `cv_scores = cross_val_score(...)`
   - Measure generalization.
- `model.fit(X_train, y_train)`
   - Train current model.
- `preds = model.predict(X_test)`
   - Test current model.
- `results[name] = {...}`
   - Save CV and test report.
- `return results`
   - Return all model results.
- `def train_best_model(...) -> tuple[str, Any, dict[str, Any]]:`
   - Define best-model trainer.
- `results = evaluate_models(features_path)`
   - Evaluate candidates.
- `best_name = max(results, key=lambda name: results[name]["cv_accuracy"])`
   - Pick highest CV accuracy.
- `model = build_models()[best_name]`
   - Rebuild chosen model.
- `df = pd.read_parquet(features_path)` etc.
   - Reload data for final training.
- `model.fit(X_train, y_train)`
   - Train best model.
- `model_path = Path(...)`
   - Choose output path.
- `model_path.parent.mkdir(...)`
   - Ensure model folder exists.
- `joblib.dump(model, model_path)`
   - Save best model.
- `return best_name, model, results`
   - Return best model info.
- `def main() -> None:`
   - Runner function.
- `best_name, _, results = train_best_model()`
   - Train best model.
- `print(f"Best model: {best_name}")`
   - Print chosen model name.
- `print(results[best_name])`
   - Print metrics.
- `if __name__ == "__main__": main()`
   - Run directly when executed.

### File: `src/data_utils.py`

- `from __future__ import annotations`
   - Future typing behavior.
- `from io import BytesIO`
   - Treat bytes like a file.
- `from pathlib import Path`
   - Path helper.
- `from typing import Any`
   - Flexible typing helper.
- `import numpy as np`, `import pandas as pd`
   - Math and tables.
- `from src.features.extract import extract_features`
   - Reuse feature extractor.
- `from src.features.pipeline import WINDOW_SIZE, window_signal`
   - Reuse windowing logic.
- `def load_latest_sensor_data(...) -> pd.DataFrame:`
   - Define CSV loading helper.
- `path = Path(...)`
   - Resolve default latest-week file.
- `if not path.exists(): raise FileNotFoundError(...)`
   - Stop if file is missing.
- `return pd.read_csv(path)`
   - Load and return sensor CSV.
- `def build_breakdown_signal(features: pd.DataFrame) -> dict[str, float]:`
   - Define fallback risk-score rule.
- `risk_score = ...`
   - Compute weighted score from RMS and Kurtosis.
- `return {...}`
   - Return risk score and status.
- `def load_cwru_npz_signal(npz_bytes, signal_key) -> tuple[np.ndarray, str]:`
   - Define raw-file loader.
- `with np.load(BytesIO(npz_bytes), allow_pickle=False) as data:`
   - Open uploaded bytes as `.npz`.
- `if not data.files:`
   - Check there is at least one array.
- `preferred_keys = ...`
   - Build preferred key list.
- `chosen_key = next(...)`
   - Choose the first matching key.
- `signal = np.asarray(...).reshape(-1)`
   - Convert chosen array into a flat float signal.
- `if signal.size == 0: raise ValueError(...)`
   - Stop on empty signal.
- `return signal, chosen_key`
   - Return signal and selected key.
- `def extract_vibration_features_from_npz(...) -> dict[str, Any]:`
   - Define raw-file-to-feature conversion.
- `signal, chosen_key = load_cwru_npz_signal(...)`
   - Load raw signal.
- `windows = window_signal(signal) if signal.size >= WINDOW_SIZE else [signal]`
   - Window long signals, or keep short signal whole.
- `feature_rows: list[dict[str, float]] = []`
   - Start feature storage.
- `for window in windows:`
   - Loop over windows.
- `if np.std(window) < 1e-6: continue`
   - Skip flat windows.
- `feature_rows.append(extract_features(window))`
   - Extract features for each valid window.
- `if not feature_rows:`
   - If all windows were skipped...
- `feature_rows.append(extract_features(signal))`
   - ...extract from whole signal as fallback.
- `features_df = pd.DataFrame(feature_rows)`
   - Convert to DataFrame.
- `feature_row = features_df.mean(...).to_dict()`
   - Average the features.
- `feature_row["signal_key"] = chosen_key`
   - Save signal name.
- `feature_row["window_count"] = len(feature_rows)`
   - Save number of used windows.
- `feature_row["raw_signal_length"] = int(signal.size)`
   - Save raw length.
- `return feature_row`
   - Return metadata + averaged features.

### File: `src/rag/ingest.py`

- `from __future__ import annotations`
   - Future typing behavior.
- `import os`
   - Read environment variables.
- `from pathlib import Path`
   - Safe file paths.
- `import chromadb`
   - Vector database library.
- `from dotenv import load_dotenv`
   - Load `.env` file.
- `from langchain_text_splitters import RecursiveCharacterTextSplitter`
   - Import text chunker.
- `from openai import OpenAI`
   - Import embedding client.
- `from pypdf import PdfReader`
   - Import PDF text reader.
- `load_dotenv()`
   - Load API key from `.env`.
- `def ingest_manual(...) -> int:`
   - Define manual ingestion function.
- Docstring block
   - Explain that this function can ingest one PDF or a whole folder.
- `pdf_path = Path(...)`
   - Choose default PDF if none is passed.
- `if not pdf_path.exists(): raise FileNotFoundError(...)`
   - Stop if PDF is missing.
- `pdf_files: list[Path] = []`
   - Prepare a list for PDFs.
- `if pdf_path.is_dir():`
   - If a folder is given...
- `pdf_files = sorted([p for p in pdf_path.glob("*.pdf")])`
   - ...find and sort all PDFs.
- `else: pdf_files = [pdf_path]`
   - Otherwise use the single file.
- `splitter = RecursiveCharacterTextSplitter(...)`
   - Create text chunking tool.
- `chroma_client = chromadb.PersistentClient(path="data/chroma_db")`
   - Open persistent Chroma database.
- `if skip_embeddings:`
   - Special path for text-only storage.
- `existing_collections = [col.name for col in chroma_client.list_collections()]`
   - List collection names.
- `if collection_name in existing_collections: chroma_client.delete_collection(collection_name)`
   - Delete old collection if needed.
- `collection = chroma_client.create_collection(..., embedding_function=None)`
   - Create new no-embedding collection.
- `else: collection = chroma_client.get_or_create_collection(..., embedding_function=None)`
   - Reuse collection if embeddings are allowed.
- `all_chunks`, `metadatas`, `ids`
   - Storage lists for chunk data.
- `for file_order, file_path in enumerate(pdf_files, start=1):`
   - Loop through each PDF.
- `reader = PdfReader(str(file_path))`
   - Open PDF.
- `for page_num, page in enumerate(reader.pages):`
   - Loop through pages.
- `text = page.extract_text() or ""`
   - Extract text safely.
- `if not text.strip(): continue`
   - Skip empty pages.
- `chunks = splitter.split_text(text)`
   - Break page into chunks.
- `for idx, chunk in enumerate(chunks):`
   - Loop over chunks.
- `all_chunks.append(chunk)`
   - Save chunk text.
- `metadatas.append({...})`
   - Save source metadata.
- `ids.append(...)`
   - Save unique chunk id.
- `if not all_chunks: raise ValueError(...)`
   - Stop if no usable text was found.
- `api_key = os.getenv("OPENAI_API_KEY")`
   - Read API key.
- `if skip_embeddings or not api_key:`
   - Use text-only path if needed.
- `print(...)`
   - Explain why embeddings are being skipped.
- `collection.add(documents=all_chunks, metadatas=metadatas, ids=ids)`
   - Store text chunks in Chroma.
- `return len(all_chunks)`
   - Return chunk count.
- `client = OpenAI(api_key=api_key, timeout=20, max_retries=1)`
   - Create embedding client with safer settings.
- `batch_size = 10`
   - Limit embedding batch size.
- `for start in range(0, len(all_chunks), batch_size):`
   - Loop over chunk batches.
- `batch = ...`, `batch_ids = ...`, `batch_metadatas = ...`
   - Slice current batch.
- `embeddings = [client.embeddings.create(...).data[0].embedding for chunk in batch]`
   - Create embeddings for each chunk.
- `collection.add(embeddings=..., documents=..., metadatas=..., ids=...)`
   - Store vectors and text.
- `except Exception as exc:`
   - Catch embedding problems.
- `print(...)`
   - Explain what failed.
- `collection.add(documents=batch, metadatas=batch_metadatas, ids=batch_ids)`
   - Fall back to text-only storage.
- `return len(all_chunks)`
   - Return total chunk count.
- `if __name__ == "__main__": print(ingest_manual())`
   - Run directly when called as a script.

### File: `src/rag/search.py`

- `from __future__ import annotations`
   - Future typing behavior.
- `import os`
   - Read environment variables.
- `from typing import Any`
   - Flexible typing helper.
- `import chromadb`
   - Vector DB library.
- `from dotenv import load_dotenv`
   - Load `.env`.
- `from openai import OpenAI`
   - OpenAI client for embeddings.
- `load_dotenv()`
   - Load API key.
- `def search_manual(query: str, k: int = 4, collection_name: str = "manual_chunks") -> dict[str, Any]:`
   - Define search function.
- `chroma_client = chromadb.PersistentClient(path="data/chroma_db")`
   - Open Chroma DB.
- `collection = chroma_client.get_or_create_collection(collection_name)`
   - Open collection.
- `api_key = os.getenv("OPENAI_API_KEY")`
   - Read API key.
- `if not api_key: raise RuntimeError(...)`
   - Stop if key is missing.
- `client = OpenAI(api_key=api_key)`
   - Create OpenAI client.
- `query_embedding = client.embeddings.create(...).data[0].embedding`
   - Turn user question into embedding.
- `return collection.query(...)`
   - Search for nearest chunks and return documents + metadata.

### File: `src/rag/llm.py`

- `from __future__ import annotations`
   - Future typing behavior.
- `from typing import Any`
   - Flexible typing helper.
- `PREDICTIVE_KEYWORDS = {...}`
   - Set of words that look like prediction questions.
- `MANUAL_KEYWORDS = {...}`
   - Set of words that look like manual questions.
- `def route_query(text: str) -> str:`
   - Define question router.
- `lowered = text.lower().strip()`
   - Normalize the text.
- `predictive_score = sum(...)`
   - Count predictive keyword matches.
- `manual_score = sum(...)`
   - Count manual keyword matches.
- `if predictive_score > manual_score and predictive_score > 0:`
   - Use predictor only when predictive signal is stronger.
- `return "predictor"`
   - Send to prediction side.
- `return "rag"`
   - Default to manual RAG side.
- `def build_rag_context(results: dict[str, Any]) -> str:`
   - Define context builder.
- `documents_list = ...`, `metadatas_list = ...`
   - Read raw retrieval outputs.
- `if not documents_list or not documents_list[0]: return ""`
   - Return empty context if nothing was found.
- `docs = documents_list[0]`
   - Take first returned document batch.
- `metas = ...`
   - Match metadata length to docs.
- `paired = []`
   - Start temporary storage.
- `for doc, meta in zip(docs, metas):`
   - Pair each chunk with metadata.
- `if not doc or not str(doc).strip(): continue`
   - Skip blank chunks.
- `file_order = ...`, `page = ...`, `chunk = ...`, `source = ...`
   - Read metadata fields safely.
- `paired.append({...})`
   - Save enriched chunk row.
- `if not any(item["meta"] for item in paired):`
   - Old compatibility path for simple results.
- `return "\n\n".join(item["doc"] for item in paired)`
   - Join plain chunks.
- `paired.sort(...)`
   - Sort by source and page order.
- `ordered_chunks: list[str] = []`
   - Start output list.
- `for item in paired:`
   - Walk through sorted chunks.
- `title = f"[{source} page {page}]"`
   - Build header label.
- `ordered_chunks.append(f"{title}\n{item['doc']}")`
   - Save labeled chunk.
- `return "\n\n".join(ordered_chunks)`
   - Return final context.
- `def build_rag_prompt(query: str, context: str, answer_style: str = "detailed") -> list[dict[str, str]]:`
   - Define prompt builder.
- `style_instruction = (...)`
   - Choose short or detailed guidance.
- `return [...]`
   - Return chat messages list.
- First message:
   - system instruction for grounded answering.
- Second message:
   - user instruction with context, question, and answer rules.
- `def extract_answer_text(response: Any) -> str:`
   - Define safe answer extractor.
- `if response is None: return ""`
   - Handle empty response.
- `if hasattr(response, "choices") and response.choices:`
   - Standard chat-completions path.
- `message = getattr(choice, "message", None)`
   - Read message safely.
- `if isinstance(message, dict): return message.get("content", "").strip()`
   - Dict response path.
- `if message is not None: return getattr(message, "content", "").strip()`
   - Object response path.
- `if hasattr(response, "output") and response.output:`
   - Alternate output-style response path.
- `output_item = response.output[0]`
   - Take first output item.
- `if isinstance(output_item, dict): return output_item.get("content", "").strip()`
   - Dict output path.
- `return str(output_item).strip()`
   - Fallback string path.
- `return ""`
   - Final safe fallback.

### File: `src/main.py`

- `from __future__ import annotations`
   - Future typing behavior.
- `from io import BytesIO`
   - Bytes-as-file helper, even though this file mostly delegates raw parsing elsewhere.
- `import os`
   - Read environment variables.
- `from typing import Any`
   - Flexible typing helper.
- `from typing import Literal`
   - Restrict answer_style values to specific strings.
- `import numpy as np`
   - Numerical helper, available for data logic.
- `import joblib`
   - Load trained model.
- `from dotenv import load_dotenv`
   - Load `.env` settings.
- `from fastapi import FastAPI, File, Form, UploadFile`
   - Build API endpoints and accept uploaded files.
- `from openai import OpenAI`
   - LLM client.
- `from pydantic import BaseModel`
   - Request schema validation.
- `from src.data_utils import (...)`
   - Import sensor and raw-file helpers.
- `from src.rag.llm import (...)`
   - Import routing, prompt, and extraction helpers.
- `from src.rag.search import search_manual`
   - Import search function.
- `load_dotenv()`
   - Load `.env` values.
- `app = FastAPI(title="Motor Pump Predictive System")`
   - Create API app.
- `client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))`
   - Create OpenAI client.
- `clf = joblib.load(...) if os.path.exists(...) else None`
   - Load best model if present.
- `SELECTED_FEATURES = [...]`
   - Define feature order for prediction.
- `def _build_sources(...)`
   - Helper to turn metadata into readable source strings.
- `sources: list[str] = []`
   - Start source list.
- `metadatas = results.get("metadatas", []) or []`
   - Read metadata safely.
- `metadata_list = metadatas[0] if metadatas else []`
   - Use first metadata batch.
- `documents = results.get("documents", []) or []`
   - Read docs safely.
- `doc_list = documents[0] if documents else []`
   - Use first doc batch.
- `for idx, _doc in enumerate(doc_list[:max_sources]):`
   - Loop over limited number of docs.
- `meta = metadata_list[idx] if idx < len(metadata_list) else {}`
   - Match metadata safely.
- `page = ...`, `source = ...`
   - Read source details.
- `if source or page: sources.append(...)`
   - Save readable source label.
- `return sources`
   - Return source list.
- `def _source_quality_from_count(source_count: int) -> str:`
   - Helper to score trust by number of sources.
- `if source_count >= 3: return "high"`
   - High quality when 3+ sources exist.
- `if source_count == 2: return "medium"`
   - Medium for 2 sources.
- `if source_count == 1: return "low"`
   - Low for 1 source.
- `return "none"`
   - No sources at all.
- `class Query(BaseModel):`
   - Request body schema for `/ask`.
- `text: str`
   - User question text.
- `answer_style: Literal["short", "detailed"] = "detailed"`
   - Answer style choice.
- `class SensorPayload(BaseModel):`
   - Request body schema for `/predict`.
- Six feature fields
   - Feature values the model needs.
- `def predict_from_feature_row(feature_row: dict[str, Any]) -> dict[str, Any]:`
   - Shared prediction helper.
- `values = [float(feature_row[name]) for name in SELECTED_FEATURES]`
   - Build ordered feature list.
- `if clf is not None:`
   - Use trained model if available.
- `proba = clf.predict_proba([values])[0]`
   - Predict class probabilities.
- `label = clf.classes_[int(proba.argmax())]`
   - Pick best class.
- `confidence = float(proba.max())`
   - Save top confidence.
- `return {...}`
   - Return prediction result.
- `except Exception:`
   - Catch model prediction failure.
- `return {...}`
   - Return user-friendly model-scoring error.
- `risk_score = ...`
   - Compute fallback risk score.
- `status = "high_risk" if risk_score > 1.0 else "normal"`
   - Convert score into status.
- `return {...}`
   - Return fallback prediction answer.
- `@app.get("/")`
   - Register health endpoint.
- `def root() -> dict[str, str]:`
   - Define root function.
- `return {"message": ...}`
   - Return status text.
- `@app.post("/ask")`
   - Register chat/manual endpoint.
- `def ask(q: Query) -> dict[str, Any]:`
   - Define ask function.
- `text = q.text`
   - Read question.
- `if route_query(text) == "predictor":`
   - Route to predictor if question smells predictive.
- `data = load_latest_sensor_data()`
   - Load sensor CSV.
- `features = data[["rms", "kurtosis"]].copy()`
   - Extract fallback rule inputs.
- `signal = build_breakdown_signal(features)`
   - Compute fallback rule answer.
- `return {...}`
   - Return predictor response.
- `except FileNotFoundError:`
   - Handle missing sensor file.
- `return {...}`
   - Tell user file is missing.
- `results = search_manual(text, k=6)`
   - Search manual with wider retrieval.
- `context = build_rag_context(results)`
   - Build combined context.
- `if not context: return {...}`
   - Handle no manual evidence.
- `api_key = os.getenv("OPENAI_API_KEY")`
   - Read API key.
- `if not api_key: return {...}`
   - Handle missing key with context preview.
- `prompt = build_rag_prompt(...)`
   - Build final prompt.
- `response = client.chat.completions.create(...)`
   - Ask LLM.
- `answer = extract_answer_text(response)`
   - Pull answer text safely.
- `if not answer: answer = ...`
   - Handle blank model output.
- `sources = _build_sources(results)`
   - Build source labels.
- `source_count = len(sources)`
   - Count source evidence.
- `source_quality = _source_quality_from_count(source_count)`
   - Score quality.
- `return {...}`
   - Return RAG answer with metadata.
- `except Exception as llm_exc:`
   - If LLM call fails...
- `preview = context[:500].strip()`
   - Take closest evidence preview.
- `return {...}`
   - Return retrieval-only fallback.
- `except RuntimeError:`
   - Catch search/runtime retrieval errors.
- `return {...}`
   - Return user-friendly fallback.
- `except Exception:`
   - Catch any remaining errors.
- `return {...}`
   - Return safe generic retry message.
- `@app.post("/predict")`
   - Register structured-feature prediction endpoint.
- `def predict(sensor: SensorPayload) -> dict[str, Any]:`
   - Define prediction endpoint.
- `feature_row = sensor.model_dump()`
   - Convert request object to dictionary.
- `return predict_from_feature_row(feature_row)`
   - Reuse prediction helper.
- `@app.post("/predict_raw")`
   - Register raw-file prediction endpoint.
- `async def predict_raw(file: UploadFile = File(...), signal_key: str = Form("DE")) -> dict[str, Any]:`
   - Define async raw upload endpoint.
- `if not file.filename.lower().endswith(".npz"):`
   - Reject non-`.npz` files.
- `return {...}`
   - Return file-type error.
- `npz_bytes = await file.read()`
   - Read uploaded file bytes.
- `feature_row = extract_vibration_features_from_npz(...)`
   - Convert raw file into averaged features.
- `prediction = predict_from_feature_row(feature_row)`
   - Predict with model or fallback.
- `return {...}`
   - Return prediction plus metadata and extracted feature values.
- `except Exception as exc:`
   - Catch raw-file processing failure.
- `return {...}`
   - Return user-friendly error.

### File: `src/app.py`

- `import streamlit as st`
   - Import Streamlit UI library.
- `import requests`
   - Import HTTP request helper.
- `st.set_page_config(...)`
   - Set page title and icon.
- `st.title(...)`
   - Show app title.
- `st.write(...)`
   - Show introduction.
- `if "history" not in st.session_state:`
   - Create history memory once.
- `st.session_state.history = []`
   - Initialize history.
- `if "manual_mode" not in st.session_state:`
   - Create manual mode state once.
- `st.session_state.manual_mode = "deep_pdf"`
   - Set default manual mode.
- `if "query" not in st.session_state:`
   - Initialize query field state.
- `st.session_state.query = ""`
   - Start with blank query.
- `if "pending_query" not in st.session_state:`
   - Initialize follow-up prefill state.
- `st.session_state.pending_query = ""`
   - Start with blank pending query.
- `mode = st.radio(...)`
   - Let user choose between Manual Chat and Sensor Prediction.
- `if mode == "Manual Chat":`
   - Start manual chat branch.
- `st.subheader(...)`
   - Manual chat heading.
- `st.write(...)`
   - Explain manual assistant behavior.
- `answer_style = st.selectbox(...)`
   - Let user choose short or detailed style.
- `if st.session_state.pending_query:`
   - Check if a follow-up prefill exists.
- `st.session_state.query = st.session_state.pending_query`
   - Move pending text into query box.
- `st.session_state.pending_query = ""`
   - Clear pending prefill.
- `with st.form("manual_chat_form", clear_on_submit=True):`
   - Create form to avoid Streamlit widget-state errors.
- `query = st.text_input("Ask a question:", key="query")`
   - Create question input box.
- `submitted = st.form_submit_button("Send")`
   - Create send button inside form.
- `cols = st.columns([0.2, 0.8])`
   - Create two layout columns.
- `if submitted and query:`
   - Only run if user pressed send and typed something.
- `q = query`
   - Save current question.
- `with st.spinner("Looking up the manual..."):`
   - Show loading spinner.
- `resp = requests.post(...)`
   - Send question to backend.
- `data = resp.json()`
   - Parse JSON response.
- `except Exception as exc:`
   - Handle network/UI call errors.
- `data = {...}`
   - Build UI-friendly error response.
- `st.session_state.history.append({...})`
   - Save question, answer, sources, preview, style, and source-quality info.
- `if cols[0].button("Clear History"):`
   - Clear chat when user clicks button.
- `st.session_state.history = []`
   - Reset history.
- `if cols[1].button("Ask a follow-up") and st.session_state.history:`
   - Prefill a follow-up question only if chat exists.
- `last_question = ...`
   - Read last question.
- `st.session_state.pending_query = ...`
   - Save follow-up prompt.
- `st.rerun()`
   - Reload app so prefill appears safely.
- `st.markdown("---")`
   - Divider line.
- `if not st.session_state.history:`
   - If no messages yet...
- `st.info(...)`
   - Show help text.
- `for i, item in enumerate(reversed(st.session_state.history)):`
   - Loop over chat messages newest first.
- `st.markdown(f"**Q:** ...")`
   - Show question.
- `st.markdown(f"**A:** ...")`
   - Show answer.
- `quality = ...`, `count = ...`, `style = ...`
   - Read metadata for badges.
- `if quality == "high": st.success(...)`
   - Show high-quality badge.
- `elif quality == "medium": st.info(...)`
   - Show medium-quality badge.
- `elif quality == "low": st.warning(...)`
   - Show low-quality badge.
- `else: st.error(...)`
   - Show no-source badge.
- `if item.get("sources"):`
   - If sources exist...
- `st.markdown("**Sources:** " + ", ".join(...))`
   - Show source list.
- `if item.get("preview"):`
   - If preview exists...
- `with st.expander("Context preview"):`
   - Create expandable section.
- `st.write(item["preview"])`
   - Show preview.
- `st.write("---")`
   - Divider between chat items.
- `else:`
   - Start Sensor Prediction branch.
- `st.subheader("Vibration Sensor Prediction")`
   - Sensor heading.
- `st.write(...)`
   - Explain sensor input choices.
- `sensor_mode = st.radio(...)`
   - Choose between raw upload and manual features.
- `if sensor_mode == "Upload raw CWRU .npz":`
   - Raw-file sub-branch.
- `uploaded = st.file_uploader(...)`
   - Upload `.npz` file.
- `signal_key = st.text_input(...)`
   - Let user choose array key.
- `if st.button("Predict from raw file") and uploaded is not None:`
   - Trigger raw prediction.
- `with st.spinner(...)`
   - Show loading spinner.
- `files = {...}`
   - Prepare multipart upload.
- `resp = requests.post(.../predict_raw...)`
   - Send raw file to backend.
- `data = resp.json()`
   - Parse backend result.
- `except Exception as exc:`
   - Handle request errors.
- `data = {...}`
   - Build UI-safe error data.
- `st.markdown(f"**Prediction result:** ...")`
   - Show prediction answer.
- `if data.get("source_file"):`
   - If backend returned file metadata...
- `st.caption(...)`
   - Show small metadata text.
- `if data.get("features"):`
   - If extracted features exist...
- `with st.expander("Extracted feature values"):`
   - Show them in an expander.
- `st.json(data["features"])`
   - Render features as JSON.
- `if data.get("prediction"):`
   - If prediction label exists...
- `prediction = str(data["prediction"])`
   - Convert label to string.
- `if prediction.lower() == "normal": st.success(...)`
   - Healthy message.
- `else: st.error(...)`
   - Fault message.
- `confidence = data.get("confidence")`
   - Read confidence.
- `if confidence is not None:`
   - Only show badge if confidence exists.
- `if confidence >= 0.8: st.success(...)`
   - High confidence badge.
- `elif confidence >= 0.6: st.info(...)`
   - Medium confidence badge.
- `else: st.warning(...)`
   - Low confidence badge.
- `with st.expander("How to read this result"):`
   - Open explanation expander.
- three `st.write(...)` lines
   - Explain healthy vs bad readings.
- `else:`
   - Manual feature-entry sub-branch.
- `st.write(...)`
   - Explain that these are model-ready features.
- six `st.number_input(...)` lines
   - Let user type feature values.
- `st.write(...)`
   - Explain these values come from raw vibration windows.
- `if st.button("Predict condition"):`
   - Trigger manual-feature prediction.
- `payload = {...}`
   - Build request body.
- `resp = requests.post("http://127.0.0.1:8000/predict", json=payload, timeout=30)`
   - Send to backend.
- `data = resp.json()`
   - Parse response.
- `except Exception as exc:`
   - Catch request errors.
- `data = {...}`
   - Build safe error response.
- `st.markdown(f"**Prediction result:** ...")`
   - Show answer.
- `if data.get("confidence") is not None:`
   - If confidence exists...
- `st.markdown(f"**Confidence:** ...")`
   - Show exact score.
- `confidence = float(data.get("confidence"))`
   - Convert to float.
- three badge lines
   - Show high/medium/low confidence badge.
- `if data.get("prediction"):`
   - If label exists...
- `prediction = str(data["prediction"])`
   - Convert to string.
- `if prediction.lower() == "normal": st.success(...)`
   - Healthy reading message.
- `else: st.error(...)`
   - Fault warning.
- explanation expander lines
   - Explain interpretation.
- `st.markdown("---")`
   - Divider line.
- `st.info(...)`
   - Show final helper text.

### File: `src/monitoring/drift.py`

- `from __future__ import annotations`
   - Future typing behavior.
- `from pathlib import Path`
   - Path helper.
- `import pandas as pd`
   - Data tables.
- `from evidently.metric_preset import DataDriftPreset`
   - Prebuilt drift metric bundle.
- `from evidently.report import Report`
   - Report builder.
- `def run_drift_check(...) -> None:`
   - Define drift check function.
- `reference_path = Path(...)`, `current_path = Path(...)`, `output_path = Path(...)`
   - Resolve default paths.
- `ref_df = pd.read_parquet(reference_path)`
   - Load reference data.
- `current_df = pd.read_parquet(current_path)`
   - Load current data.
- `report = Report(metrics=[DataDriftPreset()])`
   - Build drift report object.
- `report.run(...)`
   - Compare reference vs current data, excluding label column.
- `output_path.parent.mkdir(...)`
   - Ensure report folder exists.
- `report.save_html(output_path)`
   - Save HTML drift report.
- `print(f"Drift report saved to {output_path}")`
   - Print save location.

### File: `src/monitoring/retrain.py`

- `from __future__ import annotations`
   - Future typing behavior.
- `from pathlib import Path`
   - Path helper.
- `import pandas as pd`
   - Data tables.
- `from src.features.model_training import train_best_model`
   - Import retraining function.
- `from src.monitoring.drift import run_drift_check`
   - Import drift checker.
- `def retrain_if_needed(...) -> dict[str, object]:`
   - Define retraining decision logic.
- `reference_path = Path(...)`, `current_path = Path(...)`
   - Resolve file paths.
- `ref_df = pd.read_parquet(reference_path)`
   - Load reference features.
- `cur_df = pd.read_parquet(current_path)`
   - Load current features.
- `if ref_df.shape[0] == cur_df.shape[0] and ref_df.equals(cur_df):`
   - If same size and exact same data...
- `print("No new data detected; skipping retraining.")`
   - Explain no retraining is needed.
- `return {"status": "skipped", "reason": "no_new_data"}`
   - Return skip result.
- `run_drift_check(reference_path, current_path)`
   - Build drift report.
- `best_name, _, results = train_best_model(current_path)`
   - Retrain using current data.
- `return {...}`
   - Return retraining summary.
- `def main() -> None:`
   - CLI runner.
- `print(retrain_if_needed())`
   - Run and print result.
- `if __name__ == "__main__": main()`
   - Execute when run directly.

### File: `scripts/check_index.py`

- `import sys`, `from pathlib import Path`
   - Import path helpers.
- `sys.path.append(str(Path('.').resolve()))`
   - Make project root importable.
- `import chromadb`
   - Open Chroma library.
- `client = chromadb.PersistentClient(path='data/chroma_db')`
   - Open Chroma DB.
- `try:`
   - Start safe debug block.
- `col = client.get_or_create_collection('manual_chunks')`
   - Open manual collection.
- nested `try: count = col.count()`
   - Use normal count if supported.
- fallback `res = col.query(... include=['ids'])`
   - Try older fallback count path.
- `count = sum(...)`
   - Count fallback ids.
- deeper `except: count = 'unknown'`
   - Give up gracefully.
- `print('collection_count=', count)`
   - Show result.
- outer `except Exception as e:`
   - Catch all errors.
- `print('error checking chroma:', e)`
   - Print debug message.

### File: `scripts/check_openai.py`

- `import openai`
   - Import package itself.
- `from openai import OpenAI`
   - Import main client class.
- `with open('terminal_openai_debug.txt', 'w', encoding='utf-8') as f:`
   - Open debug output file.
- several `f.write(...)` lines
   - Write version number and whether important client attributes exist.

### File: `scripts/check_rag_query.py`

- `import json`, `import sys`, `from pathlib import Path`
   - Import file and JSON helpers.
- `sys.path.append(str(Path('.').resolve()))`
   - Make local package importable.
- `from src.rag.search import search_manual`
   - Import search function.
- `out = Path('scripts/rag_query_result.json')`
   - Choose output file.
- `try:`
   - Safe execution block.
- `results = search_manual('Where are the fuses located?', k=4)`
   - Run one retrieval query.
- `out.write_text(json.dumps(results, indent=2), encoding='utf-8')`
   - Save raw result.
- `except Exception as exc:`
   - Catch failure.
- `out.write_text(json.dumps({'error': str(exc)}, indent=2), encoding='utf-8')`
   - Save error instead.
- `print('done')`
   - Print completion marker.

### File: `scripts/chroma_debug.py`

- `import sys`, `from pathlib import Path`
   - Path/import helpers.
- `sys.path.append(str(Path('.').resolve()))`
   - Make local imports possible.
- `import chromadb`, `import inspect`
   - Import Chroma and function signature inspector.
- several `print(...)` lines at top
   - Print version and client capabilities.
- `client = chromadb.PersistentClient(path='data/chroma_db_debug')`
   - Open a debug Chroma DB.
- `print('get_or_create_collection sig:', inspect.signature(...))`
   - Show method signature.
- `try:`
   - Safe collection test.
- `col = client.get_or_create_collection('test_noembed', embedding_function=None)`
   - Create no-embedding test collection.
- `col.add(documents=['hello world'], metadatas=[{'a': 1}], ids=['doc1'])`
   - Try storing one document.
- print success or error lines
   - Show whether this approach works.

### File: `scripts/debug_ask.py`

- `import os`, `import sys`, `from pathlib import Path`
   - Standard helpers.
- `sys.path.append(str(Path(__file__).resolve().parent.parent))`
   - Add repo root for imports.
- import `search_manual`, `build_rag_context`, `route_query`, `ask`
   - Pull in main RAG helpers.
- several `print(...)` lines
   - Show cwd, root path, and API key presence.
- `query = 'How do I change the pump?'`
   - Choose debug question.
- `print('route_query', route_query(query))`
   - Show routing result.
- `try: results = search_manual(query)`
   - Run search.
- `print('search_manual ok', results.keys())`
   - Show returned keys.
- `context = build_rag_context(results)`
   - Build context.
- preview prints
   - Show context length and preview.
- `except Exception as exc:`
   - Print search error.
- nested `Q` class
   - Tiny fake object with `.text` attribute.
- `response = ask(Q(query))`
   - Call `ask()` directly.
- `print('ask response', response)`
   - Show final answer.
- error block
   - Print and re-raise ask failure.

### File: `scripts/debug_llm_flow.py`

- `import os`, `import sys`, `from pathlib import Path`
   - Standard helpers.
- `sys.path.append(str(Path(__file__).resolve().parent.parent))`
   - Make repo root importable.
- `from dotenv import load_dotenv`; `load_dotenv()`
   - Load API key.
- import search/prompt helpers and `OpenAI`
   - Needed for full LLM flow.
- `output_path = Path('terminal_llm_debug.txt')`
   - Choose debug file.
- `with output_path.open('w', encoding='utf-8') as f:`
   - Write all debug output to file.
- multiple `f.write(...)` lines
   - Log cwd, API key presence, search steps, prompt, client capabilities, response type, and final answer.
- `results = search_manual(...)`
   - Run retrieval.
- `context = build_rag_context(results)`
   - Build context.
- `prompt = build_rag_prompt(...)`
   - Build prompt.
- `client = OpenAI(...)`
   - Create client.
- `response = client.chat.completions.create(...)`
   - Call the model.
- `answer = extract_answer_text(response)`
   - Pull answer out.

### File: `scripts/probe_rag.py`

- import `search_manual` and `build_rag_context`
   - Load search tools.
- `if __name__ == "__main__":`
   - Only run when called directly.
- `query = "maintenance"`
   - Choose small test query.
- `results = search_manual(query, k=2)`
   - Retrieve top 2 chunks.
- `print("documents", len(results['documents'][0]))`
   - Print number of returned docs.
- `print(build_rag_context(results)[:1200])`
   - Print first part of combined context.

### File: `scripts/run_ingest_all.py`

- import `sys` and `Path`
   - Path helpers.
- append project root to `sys.path`
   - Make imports work.
- import `ingest_manual`
   - Pull in ingestion function.
- `if __name__ == '__main__':`
   - Run only directly.
- `count = ingest_manual('data/manuals')`
   - Ingest manual folder.
- `print('ingested_count=', count)`
   - Print chunk count.

### File: `scripts/run_ingest_capture.py`

- import `sys` and `Path`
   - Path helpers.
- add current project path to `sys.path`
   - Enable imports.
- import `ingest_manual`
   - Load ingestion function.
- `out = Path('scripts/ingest_result.txt')`
   - Choose result file.
- `try: count = ingest_manual('data')`
   - Run ingestion.
- `out.write_text(f'ingested_count={count}\n')`
   - Save success count.
- `except Exception as e:`
   - Catch failure.
- `out.write_text(f'error={e}\n')`
   - Save error text.
- `print('done')`
   - Print completion marker.

### File: `scripts/smoke_test.py`

- imports for `sys`, `Path`, `build_rag_context`, `route_query`, `extract_features`, `numpy`
   - Gather dependencies.
- add repo root to `sys.path`
   - Make local imports work.
- first two `assert` lines
   - Check routing behavior.
- third `assert`
   - Check simple chunk-join behavior.
- create `window = np.sin(...)`
   - Make fake test signal.
- `features = extract_features(window)`
   - Extract features.
- `expected = {...}`
   - Define expected keys.
- `assert set(features.keys()) == expected`
   - Verify output keys.
- `print('smoke tests passed')`
   - Print success.

### File: `scripts/test_api_flow.py`

- `from src.main import ask`
   - Import API function directly.
- class `Q`
   - Tiny object that stores `.text`.
- `if __name__ == "__main__":`
   - Run only directly.
- `print(ask(Q("How do I maintain the pump?")))`
   - Call `ask()` without HTTP.

### File: `scripts/test_ask.py`

- `import requests, json`
   - HTTP and JSON helpers.
- `url = 'http://127.0.0.1:8000/ask'`
   - API endpoint.
- `payload = {'text': 'Where are the fuses located?'}`
   - Test question.
- `try: r = requests.post(...)`
   - Send request.
- `print(r.status_code)`
   - Print HTTP status.
- nested `try:`
   - Try saving JSON response.
- write to `scripts/last_ask_response.json`
   - Save pretty JSON.
- `except Exception:`
   - If JSON save fails...
- write raw text to `scripts/last_ask_response.txt`
   - Save plain text fallback.
- outer `except Exception as e:`
   - Catch request failure.
- `print('request error:', e)`
   - Print network error.

### File: `scripts/test_e2e_endpoints.py`

- import `json`, `sys`, `Path`
   - Helpers.
- add repo root to `sys.path`
   - Enable imports.
- `from fastapi.testclient import TestClient`
   - Import local API test client.
- `from src.main import app`
   - Import app.
- `RAW_FILE = Path(...)`
   - Choose raw `.npz` test file.
- `OUT_FILE = Path(...)`
   - Choose output JSON path.
- `client = TestClient(app)`
   - Create local test client.
- `results = {}`
   - Storage dictionary.
- raw prediction block
   - If file exists, call `/predict_raw` and save status + body.
- else block
   - Save missing-file error.
- ask block
   - Call `/ask` with fuse question and save status + body.
- `OUT_FILE.write_text(...)`
   - Save combined test results.
- `print(str(OUT_FILE))`
   - Print output file path.

### File: `scripts/test_e2e_console.py`

- imports JSON/path/warnings/test client/app
   - Setup for console test.
- `warnings.filterwarnings("ignore")`
   - Hide noisy warnings.
- `client = TestClient(app)`
   - Local test client.
- `raw_file = Path(...)`
   - Choose sample raw file.
- raw prediction block
   - Test `/predict_raw` and print result preview.
- `r2 = client.post("/ask", json={"text": "predict risk from incoming sensor data"})`
   - Test `/ask` on predictor route.
- print lines
   - Print status and JSON/text preview.

### File: `scripts/test_question.py`

- import JSON and requests
   - Helpers.
- `q = "can I perform an electronic modification on this equipment"`
   - Test broad manual question.
- `r = requests.post(...)`
   - Ask live API.
- open `scripts/last_question_response.json`
   - Save response.
- `json.dump(...)`
   - Write status and body.
- `print("saved")`
   - Show completion.

### File: `scripts/test_modification_question.py`

- same import pattern as `test_question.py`
   - Uses JSON + requests.
- same question string
   - Tests modification question again.
- same request block
   - Sends live `/ask` request.
- writes to `scripts/modification_question_response.json`
   - Save response under a specific filename.
- `print("saved")`
   - Print success.

### File: `scripts/manual_random_questions_test.py`

- imports `json`, `random`, `requests`
   - Helpers for batch testing.
- `URL = "http://127.0.0.1:8000/ask"`
   - Endpoint under test.
- `QUESTIONS = [...]`
   - List of many manual-style questions.
- `random.shuffle(QUESTIONS)`
   - Change order so tests are less repetitive.
- `results = []`, `failures = []`
   - Storage lists.
- `for q in QUESTIONS:`
   - Loop through questions.
- `row = {"question": q}`
   - Start one result row.
- `resp = requests.post(...)`
   - Ask live API.
- `row["status"] = resp.status_code`
   - Save HTTP status.
- `data = resp.json() if ... else {"raw": resp.text}`
   - Parse JSON or fallback to raw text.
- several `row[...] = ...` lines
   - Save answer, route, source count, source quality, and error-flag check.
- several `if` checks
   - Mark failures for bad status, blank answer, technical error text, or LLM failure text.
- `except Exception as exc:`
   - Catch request exceptions.
- `failures.append(...)`
   - Save failure reason.
- `results.append(row)`
   - Save full row.
- `summary = {...}`
   - Build summary object.
- write `summary` to JSON file
   - Save batch test results.
- `print(json.dumps({"total": ..., "failures": ...}, indent=2))`
   - Print quick pass/fail summary.

### File: `scripts/manual_random_questions_console.py`

- imports `random`, `requests`
   - Simple helpers.
- `qs = [...]`
   - Shorter list of manual questions.
- `random.shuffle(qs)`
   - Randomize order.
- `fails = 0`
   - Start failure counter.
- `print("running", len(qs), "questions")`
   - Print test count.
- `for q in qs:`
   - Loop through questions.
- `r = requests.post(...)`
   - Send `/ask` request.
- `d = r.json()`
   - Parse JSON.
- `a = (d.get("answer") or "").strip()`
   - Clean answer text.
- `bad = (...)`
   - Decide whether answer counts as failure.
- print lines
   - Show question, status, route, source info, and answer preview.
- `if bad: print("FAIL"); fails += 1`
   - Count failures.
- exception block
   - Count request exceptions as failures.
- `print("TOTAL_FAILS", fails)`
   - Print final failure count.

### File: `tests/test_rag_llm.py`

- `from src.rag.llm import build_rag_context, route_query`
   - Import two functions to test.
- `def test_route_query_detects_predictive_requests():`
   - Define routing test.
- first `assert`
   - Confirm prediction question routes to predictor.
- second `assert`
   - Confirm maintenance question routes to RAG.
- `def test_build_rag_context_joins_document_chunks():`
   - Define context-join test.
- `results = {"documents": [["First chunk", "Second chunk"]]}`
   - Create simple fake retrieval result.
- `assert build_rag_context(results) == "First chunk\n\nSecond chunk"`
   - Confirm the function joins chunks correctly.

### File: `tests/test_feature_pipeline.py`

- `import numpy as np`
   - Import NumPy.
- `from src.features.extract import extract_features`
   - Import function under test.
- `def test_extract_features_returns_expected_keys():`
   - Define feature-output test.
- `window = np.sin(np.linspace(0, 4 * np.pi, 2048))`
   - Build a fake sine-wave signal.
- `features = extract_features(window)`
   - Run feature extraction.
- `assert set(features.keys()) == {...}`
   - Confirm all expected feature names exist.
- `assert np.isfinite(features["dominant_freq"])`
   - Ensure frequency is a real number.
- `assert np.isfinite(features["spectral_energy"])`
   - Ensure energy is a real number.

## 30. Final promise: nothing skipped

This appendix now gives a line-by-line style explanation for every Python file in the repository.

That includes:
- all main application files
- all feature-engineering files
- all RAG files
- all monitoring files
- all helper scripts in `scripts/`
- all test files in `tests/`
- the tiny root `main.py`

So this export is now the complete beginner-first walkthrough for the whole project.
