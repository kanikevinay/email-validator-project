# Email Classification Studio

A free, open-source email classification app built with Streamlit, pandas, scikit-learn, Plotly, and openpyxl.

## Project Layout

- `data/` - raw and processed datasets
- `models/` - saved model bundle and metadata
- `src/` - preprocessing, training, and prediction code
- `app.py` - Streamlit UI
- `requirements.txt` - Python dependencies

## Install

On Windows, use the Python launcher so you do not depend on `pip` being on PATH.

```bash
python -m pip install -r requirements.txt
```

If `python` is not available but `py` is, use:

```bash
py -m pip install -r requirements.txt
```

Recommended Python version: 3.11 or 3.12 for the smoothest compatibility with Streamlit and scikit-learn.

## Train the model

Your training CSV should contain one text column and one label column. Common defaults are `text` and `label`.

```bash
python -m src.train --csv-path data/train.csv --model-dir models
```

If your column names are different:

```bash
python -m src.train --csv-path data/train.csv --text-column email_body --label-column category --model-dir models
```

## Run the app

```bash
python -m streamlit run app.py
```

## Use the UI

- Single email tab: paste one email and get an instant prediction.
- Batch file tab: upload a CSV or XLSX file, choose the email column, and download an updated XLSX file with these extra columns:
  - `cleaned_email`
  - `prediction`
  - `confidence`

## Notes

- The model uses `HashingVectorizer`, so it does not build a vocabulary in memory.
- The classifier trains incrementally with `SGDClassifier(loss='log_loss')`.
- The batch UI supports Excel uploads through `openpyxl`.
