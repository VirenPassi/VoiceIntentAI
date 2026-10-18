## Overview

Intent detection aims to identify what a speaker wants to communicate from an utterance. VoiceIntentAI explores two complementary input signals:

- **Audio:** speech recordings processed into features suitable for machine-learning models.
- **Text:** the corresponding transcript, providing linguistic information.
- **Multimodal learning:** a framework for combining audio and text representations when both modalities are available.

The project uses Indonesian speech/transcript data from Mozilla Common Voice as a speech-data source. Common Voice is a speech corpus, not an intent-labelled dataset by itself; any intent labels used for training should therefore have a clearly documented source or annotation process.

## Goals

- Build a reproducible pipeline for preparing speech and text data.
- Extract useful audio features, such as MFCCs or mel-spectrograms.
- Experiment with neural models for audio and text inputs.
- Investigate whether combining modalities improves intent classification.
- Evaluate model performance and keep experiment outputs organized.

## Technical Stack

The project is designed around the following Python ecosystem. Confirm the dependencies in `requirements.txt` before installing.

| Area | Tools / libraries |
|---|---|
| Language | Python |
| Deep learning | PyTorch, torchaudio |
| Audio processing | librosa |
| Data processing | pandas, NumPy |
| Evaluation | scikit-learn |
| Text modelling | Transformers / IndoBERT, if enabled in the current implementation |

## Repository Structure

The directory names below describe the intended organization; retain or adjust entries to match the files actually present in this repository.

```text
VoiceIntentAI/
├── config/             # Experiment and model configuration
├── data/
│   ├── raw/             # Original data (usually kept out of Git)
│   └── processed/       # Cleaned data and extracted features
├── models/              # Model definitions
├── preprocessing/       # Audio and text preprocessing
├── notebooks/           # Experiments and exploratory analysis
├── utils/               # Shared utilities
├── tests/               # Tests
├── outputs/             # Local checkpoints, metrics, and predictions
├── setup_directories.py # Create the expected directory structure, if present
├── requirements.txt     # Python dependencies
└── README.md
```

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/VirenPassi/VoiceIntentAI.git
cd VoiceIntentAI
```

### 2. Create a virtual environment

**Windows (PowerShell):**

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**Linux / macOS:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

If the repository contains `requirements.txt`:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

PyTorch installation can vary by operating system and hardware. If the requirements file does not specify a compatible build, use the official [PyTorch installation selector](https://pytorch.org/get-started/locally/).

### 4. Prepare the dataset

1. Obtain the Indonesian speech data from [Mozilla Common Voice](https://commonvoice.mozilla.org/).
2. Review the dataset's current terms and conditions before using or redistributing it.
3. Place data in the expected local data directory and run the preprocessing scripts available in the repository.
4. Document how intent labels were created or sourced. Do not treat ordinary speech transcripts as intent labels without a separate labelling step.

Dataset files, personal recordings, and other large or restricted assets should generally remain outside version control.

## Running the Project

The exact commands depend on the scripts present in the repository. Inspect the available entry points before running training or evaluation. If `setup_directories.py` is included, the expected directory setup may be created with:

```bash
python setup_directories.py
```

Use the actual training and evaluation scripts from the repository for subsequent steps. Add the verified commands here once their filenames and arguments have been confirmed; this README intentionally does not invent script names or claim that an unverified command works.

## Model Approach

The project explores a multimodal pipeline with the following conceptual stages:

1. **Audio preprocessing** — load recordings, standardize audio where needed, and extract acoustic features such as MFCCs or mel-spectrograms.
2. **Audio representation** — experiment with a neural audio model, such as a CNN combined with a bidirectional LSTM, if present in the implementation.
3. **Text representation** — encode transcripts using an Indonesian-language transformer such as IndoBERT, if configured.
4. **Fusion** — combine audio and text representations using an appropriate fusion mechanism, such as attention or gated fusion, if implemented.
5. **Prediction and evaluation** — produce intent predictions and assess them using held-out data and suitable classification metrics.

This section describes the intended experimental direction, not a guarantee that every component is implemented in the current codebase.

## Evaluation and Reporting

For a meaningful evaluation, document:

- How intent classes were defined and labelled.
- Dataset splits and whether speakers overlap between training and test sets.
- Class distribution and any imbalance-handling strategy.
- Metrics such as accuracy, precision, recall, F1-score, and a confusion matrix.
- Random seeds, model configuration, and the evaluation procedure.

Report test results only when they can be reproduced from the committed code and documented data-processing steps. Avoid presenting a high score without explaining the labels, data split, and possible leakage risks.

## Data, Models, and Git Hygiene

Do not commit the following unless there is a clear, appropriate reason:

- API keys, credentials, `.env` files, or personal information.
- Virtual environments (`.venv/`, `venv/`) and Python cache files.
- Large raw datasets or downloaded speech archives.
- Large model checkpoints and generated outputs that can be reproduced.

Consider using a `.gitignore` for these files and Git LFS or an appropriate model/data hosting service for assets that genuinely need to be shared. Keep dataset attribution and licence information with the project.

## Current Limitations

- Common Voice provides speech and transcript data, but it does not automatically provide the intent labels required for intent classification.
- Results depend on the quality and provenance of intent annotations, dataset splits, and preprocessing.
- Audio-only, text-only, and multimodal models should be compared using the same evaluation protocol before drawing conclusions.
- Hardware requirements depend on the selected models and whether GPU acceleration is available.

## Roadmap

- [ ] Document the intent-label schema and annotation process.
- [ ] Make preprocessing and train/evaluation commands reproducible.
- [ ] Add tests for data loading and preprocessing.
- [ ] Compare audio-only, text-only, and multimodal baselines.
- [ ] Publish reproducible evaluation metrics and experiment configurations.
- [ ] Document dataset provenance, licences, and model limitations.

## Contributing

Contributions and constructive feedback are welcome. For a change, please describe the motivation, keep experiments reproducible, and avoid committing private data, credentials, or large generated files.

## Author

**Viren Passi**  
GitHub: [@VirenPassi](https://github.com/VirenPassi)

---

*This README should be finalized against the actual repository contents before release. In particular, verify the folder tree, dependency list, model components, and runnable commands.*
tart them with some minimal added configuration and they go. Each of these too are exposing only the ports needed to do their job on the network and nothing else. So a hacker can’t typically talk to them like a regular computer.

Now I am building an app that communicates with these two containers, putting data in and ou on their network interfaces.

When time comes to deploy my app to production, I don’t need to set up separate servers and what not. I just save my containers into image files then just go to Amazon and deploy with docker to the server instances I choose. I don’t care what operating system Amazon is running on those servers, because docker will build me containers that are identical to those I saved on my own computer and run them.

So it could be that Amazon I running Amazon linux but my container is running fedora, or Ubuntu, doesn’t matter, docker will handle the translation between the container and the host.

If get pissed off at Amazon and decide to go to Google cloud, I deploy the same files there, I don’t care what Google is using to run their servers, docker will do the translation.

Unlike virtual machines, because docker create these light weight containers, I won’t loose much performance.

Finally, a docker file is a simple instruction file that tells docker to download an image, then run some commands on it, such as install additional software, etc
