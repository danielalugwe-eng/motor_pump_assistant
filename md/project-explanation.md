# Motor Pump Predictive System — Simple Explanation

## What this project does
This project has two assistants:
- A **chat assistant** that reads a machine manual and answers questions about it.
- A **sensor prediction assistant** that takes machine vibration features and predicts whether the pump is healthy or at risk.

It also includes a little app so you can type questions or enter sensor values.

## How to use it
1. Start the API server:
   ```powershell
   uv run uvicorn src.main:app --reload --host 127.0.0.1 --port 8000
   ```
2. Start the UI app:
   ```powershell
   uv run streamlit run src/app.py
   ```
3. Open the Streamlit page in your browser.

## What files are important

### `src/main.py`
This is the API server.
- It reads your question.
- If the question sounds like a sensor prediction question, it tries to use the prediction model or sensor data.
- Otherwise, it searches the manual using the RAG pipeline and asks the OpenAI model to answer.
- It also exposes a new endpoint, `/predict`, to take raw sensor feature values and return a model prediction.
- It also exposes `/predict_raw`, which accepts a raw CWRU `.npz` vibration file, extracts features from the signal, and predicts from those features.

### `src/app.py`
This is the user interface.
- It has two modes: `Manual Chat` and `Sensor Prediction`.
- In `Manual Chat`, you type a question and it sends the question to `/ask`.
- In `Sensor Prediction`, you can upload a raw CWRU `.npz` file or type the extracted vibration features manually.
- It sends raw files to `/predict_raw` and feature values to `/predict`.

### `src/data_utils.py`
This helps with sensor data.
- `load_latest_sensor_data()` reads a CSV file called `data/processed/latest_week.csv`.
- `build_breakdown_signal()` turns two features into a simple risk score.
- `load_cwru_npz_signal()` reads the raw vibration array from a CWRU `.npz` file.
- `extract_vibration_features_from_npz()` turns a raw vibration file into the same model-ready features used during training.

### `src/rag/ingest.py`
This is the RAG ingest script.
- It reads one PDF file or all PDFs in a folder.
- It chops the text into small chunks.
- Each chunk is stored in ChromaDB along with metadata like file name, page, chunk index, and the order of the file.
- It can create embeddings using OpenAI or store text-only chunks if embeddings are unavailable.

### `src/rag/search.py`
This is the RAG search script.
- It takes a user question.
- It converts the question into an OpenAI embedding.
- It asks ChromaDB for the closest manual chunks.
- It returns the matching document snippets and metadata.

### `src/rag/llm.py`
This builds the prompt and answer logic.
- `route_query()` decides whether a question should go to the manual chat or the predictor.
- `build_rag_context()` orders the returned chunks by file, page, and chunk.
- `build_rag_prompt()` builds a prompt asking the AI to answer using only manual text.
- `extract_answer_text()` gets the final answer out of the OpenAI response.

`route_query()` now uses expanded keyword sets instead of a tiny list, so broad question styles are handled better. It checks many predictive terms (like fault, anomaly, condition monitoring, vibration trend, remaining useful life) versus many manual/procedure terms (like modification, electrical, wiring, maintenance, replace, safety, panel) and routes to the best path.

### `src/features/extract.py`
This computes features from vibration windows.
- `extract_features(window)` calculates things like mean, RMS, kurtosis, spectral energy, and dominant frequency.
- These are the numbers the prediction model uses.

This is the bridge from raw vibration to prediction. The app does not use raw waveforms directly; it converts them into the exact feature numbers the model understands.

### `src/features/pipeline.py`
This builds the training data.
- It reads CWRU `.npz` files.
- It cuts long vibration signals into smaller windows.
- It computes features for each window.
- It writes the feature table to `data/processed/features.parquet`.

This is why the predictor is rooted in the actual CWRU vibration dataset. The same kind of raw signal that lives in `data/raw/CWRU_Bearing_NumPy/Data` is windowed and feature-extracted before training or prediction.

### `src/features/model_training.py`
This trains the sensor fault model.
- It builds several candidate models: random forest, gradient boosting, and SVM.
- It evaluates them using cross-validation.
- It chooses the best model and saves it to `models/best_fault_classifier.pkl`.

### `src/monitoring/drift.py` and `src/monitoring/retrain.py`
These are for later use.
- Drift monitoring watches whether new data starts to look different from training data.
- Retraining will let you rebuild the model when the machine behavior changes.

### `scripts/*.py`
These are helper scripts for testing and debugging.
- `scripts/test_ask.py`: sends a question to `/ask`.
- `scripts/run_ingest_all.py`: runs the ingest function on a folder.
- `scripts/check_index.py`: inspects the ChromaDB collection.
- `scripts/debug_llm_flow.py`: tests the LLM path.

## How the RAG pipeline works, like you are 5
1. The manual is a big book.
2. We cut the book into small pieces.
3. We turn each piece into a secret number code that means what it says.
4. We store those secret numbers in a smart library called ChromaDB.
5. When you ask a question, we turn your question into another secret number code.
6. We look in the library for the pieces whose secret codes are closest to the question code.
7. We take those pieces and give them to the AI.
8. The AI answers using only those pieces.

That means the assistant is not guessing from memory; it is looking up the manual first.

## How the LLM works in this project (Case Study, like you are 5)
Think of your assistant like a smart kid in a library:

1. You ask a question, like "Where is the fuse located?"
2. The kid does **not** answer from memory first.
3. The kid runs to the manual shelf (your ChromaDB index) and finds the closest pages.
4. The kid comes back with those page snippets only.
5. Then the kid asks a teacher (the LLM) to explain the answer using those snippets.
6. The teacher must show where the answer came from (source pages), not just guess.

In your code, this happens in order:

- `src/rag/search.py`:
   - turns your question into an embedding
   - retrieves the nearest manual chunks from ChromaDB

- `src/rag/llm.py`:
   - `build_rag_context()` arranges chunks and includes page labels
   - `build_rag_prompt()` tells the LLM to answer directly from evidence
   - `extract_answer_text()` safely reads the answer from the OpenAI response

- `src/main.py` (`/ask` endpoint):
   - calls search
   - builds context
   - sends prompt to the chat model (`gpt-4o-mini`)
   - returns `answer`, `context_preview`, and `sources`

### Why this is powerful
- The LLM can answer many styles of questions ("where", "how", "what", "why")
- But it stays grounded in your PDF manual instead of making things up
- If exact details are missing, it gives the closest supported guidance and asks one focused follow-up

### Example from your project
Question: "Where is the fuse located?"

What your RAG LLM should do:
1. Retrieve chunks mentioning fuses and electric panel
2. Answer directly (for example: in the electric panel / power board area)
3. Show supporting source pages like `[top_ex.pdf page 39]`

That is exactly what "RAG + LLM" means in this case study: **find evidence first, explain second**.

### What happens for unusual or broad questions
Sometimes a user asks a policy-style or wide question, such as:
- "Can I perform an electronic modification on this equipment?"

The system now handles this better by:
1. Sending the question to the manual RAG path (not the sensor predictor path).
2. Retrieving more manual chunks for wider coverage.
3. Letting the LLM answer directly if evidence exists.
4. Falling back to retrieval-only evidence if the LLM call fails.
5. Returning a user-friendly response even when retrieval cannot complete, instead of showing raw technical errors.

So even when the question is broad, the assistant still returns the closest manual-grounded answer instead of a hard error.

### Expanded keyword routing (so more questions work)
The question router now recognizes many more words and phrases.

- Predictor-oriented keywords include words like: `predict`, `fault`, `failure`, `risk`, `sensor`, `vibration`, `anomaly`, `condition monitoring`, `remaining useful life`.
- Manual-oriented keywords include words like: `electrical`, `electronic`, `modification`, `wiring`, `panel`, `maintenance`, `replace`, `safety`, `can I`, `allowed`, `permission`, `terminal`.

This wider keyword map reduces misrouting and helps the app answer a broader range of real user questions.

## What each library does

| Library | What it does in this project |
|---|---|
| `fastapi` | Makes the API server that answers questions. |
| `uvicorn` | Runs the API server. |
| `streamlit` | Builds the app interface you open in the browser. |
| `openai` | Talks to OpenAI for embeddings and chat answers. |
| `chromadb` | Stores manual chunks and finds matching text by meaning. |
| `pypdf` | Reads text out of PDF files. |
| `langchain-text-splitters` | Splits PDF text into chunks for RAG. |
| `requests` | Lets the Streamlit app call the FastAPI server. |
| `pandas` | Works with tables of sensor data and features. |
| `numpy` | Does numerical math for vibration signals. |
| `scikit-learn` | Builds and evaluates the sensor prediction model. |
| `scipy` | Computes spectral and statistical features from vibration. |
| `joblib` | Saves and loads the trained model. |
| `python-dotenv` | Loads the `.env` file with your OpenAI key. |
| `python-multipart` | Lets FastAPI receive uploaded raw `.npz` files. |
| `matplotlib` / `seaborn` | For charts and plots when you want to visualize data. |
| `evidently` | Monitors data drift over time. |

## How the sensor prediction works, like you are 5
1. The machine makes a sound signal when it runs.
2. We turn that sound into numbers like how bumpy it is, how spiky it is, and what note it mostly plays.
3. The model learns from many examples of healthy and broken bearings.
4. When you give raw vibration data, the app cuts it into windows and turns it into the same numbers the model learned from.
5. When you give the numbers or the raw file, the model says "healthy" or "not healthy".

In the app, you can upload a raw `.npz` file from the CWRU dataset or type the extracted numbers directly for a quick prediction.

## How to know if a reading is healthy or bad
The predictor decides this from the vibration features it extracts from the raw signal.

- **Healthy reading** means the model predicts `Normal` or the fallback score stays in the low-risk range.
- **Bad reading** means the model predicts a fault class like `InnerRace`, `Ball`, or `OuterRace`, or the fallback score is in the high-risk range.
- **Low-risk signs** usually look like smaller `RMS`, lower `Kurtosis`, and a smoother signal with fewer sharp spikes.
- **Bad-reading signs** usually look like higher `RMS`, higher `Kurtosis`, larger `Crest Factor`, and more sudden vibration spikes.
- If you upload a raw CWRU `.npz` file, the app turns that vibration into windows, extracts features, and then predicts the condition from those features.
- If the model is confident, it shows a higher confidence score. If the confidence is low or the fallback rule is used, treat the result as a warning and check the raw vibration data again.

Simple rule of thumb:
- `Normal` = healthy reading
- `InnerRace`, `Ball`, `OuterRace` = bad reading

## What I changed for you
- Added a Streamlit UI with a manual chat mode and a sensor prediction mode.
- Added a `/predict` endpoint in `src/main.py`.
- Added a `/predict_raw` endpoint in `src/main.py` for raw CWRU vibration files.
- Updated `src/rag/search.py` so it no longer asks ChromaDB for unsupported `ids` in query results.
- Updated `src/rag/llm.py` so it builds context from manual chunks in proper order.
- Updated the sensor UI so it can accept raw vibration files from `data/raw/CWRU_Bearing_NumPy/Data`.
- Added `md/project-explanation.md` with this full explanation.

## How to run the full thing now
1. Make sure `.env` contains your `OPENAI_API_KEY`.
2. Start the backend:
   ```powershell
   uv run uvicorn src.main:app --reload --host 127.0.0.1 --port 8000
   ```
3. Start the UI:
   ```powershell
   uv run streamlit run src/app.py
   ```
4. Open the browser page Streamlit gives you.

If you want, I can also update `README.md` with a short quickstart and wiring instructions from this explanation.