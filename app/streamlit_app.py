from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

import sklearn
import joblib
import sys
import streamlit as st

st.write("Python:", sys.version)
st.write("Scikit-learn:", sklearn.__version__)
st.write("Joblib:", joblib.__version__)

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
MODELS_DIR = ROOT / "models"
DATA_FP = SRC_DIR / "feature_engineered_album.csv"
REG_MODEL_FP = MODELS_DIR / "tuned_regression_model.pkl"
CLF_MODEL_FP = MODELS_DIR / "tuned_classification_model.pkl"


@st.cache_data
def load_dataset() -> pd.DataFrame:
	return pd.read_csv(DATA_FP)


@st.cache_resource
def load_model(model_path: Path):
	if model_path.exists():
		return joblib.load(model_path)
	return None


def make_input_row(df: pd.DataFrame, album_type: str, bpm: int, avg_ms_played: float, avg_percent: float, release_year: int) -> pd.DataFrame:
	row = {
		"release_year": release_year,
		"bpm": bpm,
		"average_ms_played": avg_ms_played,
		"avg_percent_song_played": avg_percent,
		"song_age_years": max(0, 2026 - release_year),
		"average_played_sec": avg_ms_played / 1000.0,
		"bpm_x_completion": bpm * avg_percent,
		"album_type": album_type,
	}
	return pd.DataFrame([row])


st.set_page_config(page_title="Spotify Popularity Dashboard", layout="wide")
st.title("Spotify Song Popularity Prediction & Listening Behavior")
st.caption("Regression first, then classification and behavior analytics.")

if not DATA_FP.exists():
	st.error(f"Missing engineered dataset: {DATA_FP}")
	st.stop()

df = load_dataset()
reg_model = load_model(REG_MODEL_FP)
clf_model = load_model(CLF_MODEL_FP)

if reg_model is None:
	st.warning(f"Regression model not found at {REG_MODEL_FP}")
if clf_model is None:
	st.warning(f"Classification model not found at {CLF_MODEL_FP}")

tab_overview, tab_eda, tab_predict = st.tabs(["Overview", "EDA", "Predict"])

with tab_overview:
	col1, col2, col3, col4 = st.columns(4)
	col1.metric("Tracks", f"{len(df):,}")
	col2.metric("Median Streams", f"{df['number_of_streams'].median():.0f}")
	col3.metric("Mean BPM", f"{df['bpm'].mean():.1f}")
	col4.metric("Popular Rate", f"{(df['is_popular'].mean() * 100):.1f}%")

	st.subheader("Top Tracks by Streams")
	top = df.nlargest(10, "number_of_streams")[["track_name", "number_of_streams"]]
	fig_top = px.bar(top.sort_values("number_of_streams"), x="number_of_streams", y="track_name", orientation="h")
	st.plotly_chart(fig_top, use_container_width=True)

with tab_eda:
	c1, c2 = st.columns(2)
	with c1:
		st.subheader("Stream Distribution")
		fig_hist = px.histogram(df, x="number_of_streams", nbins=40)
		st.plotly_chart(fig_hist, use_container_width=True)

	with c2:
		st.subheader("BPM vs Streams")
		fig_scatter = px.scatter(
			df,
			x="bpm",
			y="number_of_streams",
			color="album_type",
			hover_data=["track_name"],
			opacity=0.7,
		)
		st.plotly_chart(fig_scatter, use_container_width=True)

	st.subheader("Average Streams by Album Type")
	grp = df.groupby("album_type", as_index=False)["number_of_streams"].mean().sort_values("number_of_streams", ascending=False)
	fig_album = px.bar(grp, x="album_type", y="number_of_streams")
	st.plotly_chart(fig_album, use_container_width=True)

with tab_predict:
	st.subheader("Predict From Song Attributes")

	album_type_options = sorted(df["album_type"].dropna().unique().tolist())
	col1, col2, col3 = st.columns(3)
	with col1:
		album_type = st.selectbox("Album Type", album_type_options, index=0)
		bpm = st.slider("BPM", min_value=60, max_value=220, value=int(df["bpm"].median()))
	with col2:
		avg_ms_played = st.number_input("Average ms played", min_value=10_000.0, max_value=600_000.0, value=float(df["average_ms_played"].median()))
		avg_percent = st.slider("Average % song played", min_value=30.0, max_value=100.0, value=float(df["avg_percent_song_played"].median()))
	with col3:
		release_year = st.slider("Release year", min_value=int(df["release_year"].min()), max_value=2026, value=int(df["release_year"].median()))

	input_df = make_input_row(df, album_type, bpm, avg_ms_played, avg_percent, release_year)
	st.dataframe(input_df, use_container_width=True)

	pred_col1, pred_col2 = st.columns(2)
	with pred_col1:
		st.markdown("### Regression")
		if reg_model is not None:
			pred_streams = float(reg_model.predict(input_df)[0])
			st.success(f"Predicted streams: {pred_streams:.1f}")
		else:
			st.info("Regression model unavailable.")

	with pred_col2:
		st.markdown("### Classification")
		if clf_model is not None:
			pred_class = int(clf_model.predict(input_df)[0])
			label = "Popular" if pred_class == 1 else "Not Popular"
			st.success(f"Predicted class: {label}")
		else:
			st.info("Classification model unavailable.")
