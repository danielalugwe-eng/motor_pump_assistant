# Dataset Foundation and Model Testing Guide

This guide explains the dataset in more depth and shows how to think about healthy versus faulty motor-pump readings.

## What dataset is this project using?

This project is built around the CWRU bearing vibration dataset. The dataset contains vibration recordings of bearings under different conditions.

The labels in the repository include:
- Normal: healthy condition
- InnerRace: inner race fault
- Ball: ball fault
- OuterRace: outer race fault

The files are stored in .npz format, which is a NumPy-compressed archive. Inside each file, the vibration signal is stored as an array such as DE or FE.

### Why this matters

The project is not yet using a full commercial pump dataset. Instead, it uses a classic benchmark dataset to teach the model the difference between healthy vibration patterns and fault-like vibration patterns.

That makes this a strong foundation for learning, but for a real factory deployment you would eventually want:
- your own pump or motor data,
- labels from real maintenance events,
- a larger and more realistic operating range.

## How to read the raw vibration signal

A vibration signal is just a long list of numbers sampled over time.

In the code, the signal is read from the .npz file and then split into windows.

Why windowing is important:
- one long recording is too large to learn from directly,
- short chunks make the data easier to analyze,
- each window can be treated as one example for the model.

The project uses windows of size 2048 with 50% overlap.

This means:
- a chunk of vibration is analyzed at a time,
- overlap keeps more information near the edges of each chunk,
- the model sees many slightly different examples from one long signal.

## What the model learns from

The model does not look at the raw vibration waveform directly. Instead, it learns from features.

The key features are:

| Feature | What it means | Why it often helps detect faults |
|---|---|---|
| mean | average vibration level | mostly a sanity check |
| std | spread of the signal | larger spread often means more violent vibration |
| rms | overall amplitude strength | higher in stronger vibration |
| skew | asymmetry of the signal | fault impulses may create one-sided spikes |
| kurtosis | how sharp the spikes are | very important for impulsive bearing faults |
| ptp | peak-to-peak amplitude | shows how extreme the signal became |
| crest_factor | peak compared to overall vibration | high when sharp impacts stand out |
| dominant_freq | strongest frequency component | changes with fault type and geometry |
| spectral_energy | total energy across frequencies | higher energy usually means stronger vibration |
| spectral_entropy | complexity of the frequency pattern | faults often make the spectrum more irregular |

## How to tell a healthy vs bad reading

A healthy reading usually looks like this:
- lower overall vibration energy,
- smoother signal,
- fewer sharp spikes,
- lower kurtosis and crest factor,
- less extreme peak values.

A bad or fault-like reading usually looks like this:
- higher RMS and std,
- larger peak-to-peak range,
- stronger impulses or impacts,
- higher kurtosis,
- higher crest_factor,
- more irregular frequency content.

In plain words:
- A healthy motor pump reading is usually calmer and more regular.
- A bad reading often feels more impulsive and energetic.

## Practical interpretation for this project

Because the model is trained on vibration features, you should evaluate a new reading in two ways:

1. Look at the feature values directly.
2. Let the trained model predict the health status.

### A simple rule of thumb

If a new window has:
- high RMS,
- high kurtosis,
- high crest_factor,
- high spectral_energy,

then it is more likely to look fault-like.

If it has:
- low RMS,
- low kurtosis,
- low crest_factor,
- smoother spectrum,

then it is more likely to look healthy.

## How the model is tested in this repository

The training pipeline uses several steps:

1. Split the dataset into training and testing sets.
2. Train the model on the training set.
3. Test it on unseen data.
4. Evaluate using cross-validation and a classification report.

The code uses:
- train_test_split() to hold out part of the data
- StratifiedKFold and cross_val_score() to estimate generalization
- classification_report() to show precision, recall, and F1 score

### What a good result looks like

A model is doing well when:
- it performs well on the held-out test set,
- it is consistent across cross-validation folds,
- it correctly identifies both healthy and faulty patterns.

## How to test a new pump reading

Use this workflow:

1. Load the new vibration signal.
2. Split it into windows.
3. Extract features from each window.
4. Run the model on those features.
5. Review the predictions over many windows.
6. Decide whether the machine looks healthy or suspicious.

A good practice is to aggregate many windows instead of trusting one single chunk.

If 70% or more of the windows are predicted as fault-like, that is a stronger signal than a single isolated window.

## Example interpretation checklist

When you inspect a new reading, ask:

- Does the signal contain sharp impulses?
- Are the peaks unusually large?
- Does the spectrum look more complex than normal?
- Are the feature values clearly above the healthy baseline?
- Does the model consistently predict fault-like behavior across multiple windows?

If the answer is yes to several of these, the reading is likely suspicious.

## Important warning

One single window can be noisy or misleading. A real diagnosis should consider:
- multiple windows,
- several features together,
- trends over time,
- sensor quality and operating conditions.

A healthy machine can show a noisy window by chance, and a faulty machine can look normal for a short moment.

That is why the project uses many windows and evaluates the whole set of features rather than one value alone.

## Recommended next step

The best next step is to collect real pump data with known outcomes and then retrain the model on that data. That will make the system much more meaningful for your actual motor-pump environment.
