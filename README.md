# Email Validation and Classification Workspace

This repository contains two Streamlit applications:

- `email_verifier`: validates email addresses in bulk using syntax checks and MX lookups.
- `email_classifier`: classifies email text into categories with a scikit-learn model.

The workspace also includes root launcher scripts so you can run both tools from the repository root.

## Repository Structure

- `email_verifier/` - bulk email verification app and validation logic
- `email_classifier/` - model training, prediction pipeline, and classifier app
- `app.py` - root launcher for the verifier UI
- `email_verifier_app.py` - alternate root launcher for the verifier UI
- `yourscript.py` - alternate root launcher for the verifier UI
- `train.py` - root launcher for classifier training
- `predict.py` - root launcher for classifier prediction
- `requirements.txt` - root dependency entry point (currently installs verifier requirements)

## Prerequisites

- Python 3.10+
- pip
- Git

## Setup

### 1) Clone and move into the project

```bash
git clone https://github.com/kanikevinay/email-validator-project.git
cd email-validator-project
```

### 2) Create and activate a virtual environment (recommended)

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3) Install dependencies

Verifier dependencies from the root:

```bash
python -m pip install -r requirements.txt
```

Classifier dependencies:

```bash
python -m pip install -r email_classifier/requirements.txt
```

## Run the Email Verifier App

From the repository root:

```bash
python -m streamlit run app.py
```

Equivalent alternatives:

```bash
python -m streamlit run email_verifier_app.py
python -m streamlit run yourscript.py
python -m streamlit run email_verifier/app.py
```

### Verifier Output Files

Generated outputs are written to:

- `email_verifier/data/working_emails.csv`
- `email_verifier/data/invalid_emails.csv`
- `email_verifier/data/unverified_emails.csv`
- temporary uploaded files under `email_verifier/data/uploads/`

If DNS lookup is unavailable, syntax-valid addresses are written to the unverified file instead of being mislabeled as invalid.

## Train the Email Classifier

Option A: from repository root via launcher:

```bash
python train.py --csv-path email_classifier/data/sample_train.csv --model-dir email_classifier/models
```

Option B: inside classifier project folder:

```bash
cd email_classifier
python -m src.train --csv-path data/sample_train.csv --model-dir models
```

## Run the Email Classifier App

```bash
cd email_classifier
python -m streamlit run app.py
```

## Run Classifier Predictions from CLI

From repository root:

```bash
python predict.py --help
```

From classifier folder:

```bash
cd email_classifier
python -m src.predict --help
```

## Notes

- The verifier is optimized for large batches with chunked processing and asynchronous DNS/MX checks.
- The classifier uses a `HashingVectorizer` with an `SGDClassifier` (`log_loss`) and supports batch file prediction in the UI.

## License

Add your preferred license information here (for example: MIT).
