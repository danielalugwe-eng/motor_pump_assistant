# Project Documentation Index

This folder collects the beginner-friendly documentation for the predictive maintenance project.

## What this documentation covers

- A plain-English explanation of every major folder in the repository
- A script-by-script description of what each file does and why it exists
- A foundation guide to the vibration dataset and how to tell a healthy signal from a damaged one
- A practical workflow for testing the predictive model on new pump-like readings

## Suggested reading order

1. Start with the folder and script guide to understand the project structure.
2. Read the dataset and model-testing guide to learn how the model recognizes healthy versus faulty vibration.
3. Use the commands in this folder as a checklist when you run the project locally.

## Quick start commands

```bash
uv run python -m src.features.run_pipeline
uv run python -m src.features.model_training
uv run uvicorn src.main:app --reload
```

## Main ideas behind the system

This repository is a small predictive maintenance workflow. It learns from vibration signals, turns them into meaningful features, trains a classifier, and then looks for signals that suggest a machine is moving toward a fault state.

The core idea is simple:

- Healthy machines usually produce smoother and more stable vibration patterns.
- Faulty machines often produce sharper impulses, stronger overall vibration, and different frequency content.
- A model can learn this difference from historical labeled examples.

## Files in this documentation folder

- [folder-and-script-guide.md](folder-and-script-guide.md)
- [model-testing-and-dataset-guide.md](model-testing-and-dataset-guide.md)
