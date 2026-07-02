# Spotify Song Popularity Prediction & Listening Behavior Analysis

Project scaffold for predicting Spotify song popularity and analyzing listening behavior.

Data
- Place the CSV files (`spotify_skz_counts_album.csv`, `spotify_skz_counts_day.csv`, `spotify_skz_streaming_4.25_4.26.csv`) into the `data/` directory (or update the paths in the notebooks).

Getting started

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook
```

Structure
- `data/` — dataset CSVs (not included)
- `notebooks/` — EDA and modeling notebooks
- `src/` — reusable scripts
- `models/` — saved models
- `app/` — Streamlit app for demo

Next steps
- Run `notebooks/01-data-exploration.ipynb` to inspect data and run cleaning/EDA.
- Implement pipeline in `src/` and save best model to `models/`.
