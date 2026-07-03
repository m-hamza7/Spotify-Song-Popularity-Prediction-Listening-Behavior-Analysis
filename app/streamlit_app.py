from pathlib import Path
import sys

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
	sys.path.insert(0, str(SRC_DIR))
MODELS_DIR = ROOT / "models"
DATA_FP = SRC_DIR / "feature_engineered_album.csv"
REG_MODEL_FP = MODELS_DIR / "tuned_regression_model.pkl"
CLF_MODEL_FP = MODELS_DIR / "tuned_classification_model.pkl"
REC_MODEL_FP = MODELS_DIR / "track_recommender.pkl"


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
st.caption("Regression, classification, behavior analytics, and track recommendations.")

if not DATA_FP.exists():
	st.error(f"Missing engineered dataset: {DATA_FP}")
	st.stop()

df = load_dataset()
reg_model = load_model(REG_MODEL_FP)
clf_model = load_model(CLF_MODEL_FP)
rec_model = load_model(REC_MODEL_FP)

if reg_model is None:
	st.warning(f"Regression model not found at {REG_MODEL_FP}")
if clf_model is None:
	st.warning(f"Classification model not found at {CLF_MODEL_FP}")
if rec_model is None:
	st.warning(f"Recommendation model not found at {REC_MODEL_FP}. Run `python src/train_recommendation.py`.")

tab_overview, tab_eda, tab_predict, tab_recommend = st.tabs(
	["Overview", "EDA", "Predict", "Recommend"]
)

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

with tab_recommend:
	st.subheader("Track Recommendations")
	st.caption(
		"Content-based recommendations using BPM, album type, listening completion, and release year."
	)

	if rec_model is None:
		st.info("Train the recommender with `python src/train_recommendation.py` to enable this tab.")
	else:
		mode = st.radio(
			"Recommendation mode",
			["Similar to a track", "Based on listening profile"],
			horizontal=True,
		)
		n_recs = st.slider("Number of recommendations", min_value=3, max_value=15, value=5)

		if mode == "Similar to a track":
			track_options = sorted(df["track_name"].dropna().unique().tolist())
			seed_track = st.selectbox("Choose a seed track", track_options)
			if st.button("Get recommendations", type="primary"):
				try:
					recs = rec_model.recommend_by_track(seed_track, n=n_recs)
					st.success(f"Tracks similar to **{seed_track}**")
					st.dataframe(recs, use_container_width=True, hide_index=True)

					fig_recs = px.bar(
						recs,
						x="similarity",
						y="track_name",
						orientation="h",
						hover_data=["album_name", "bpm", "number_of_streams"],
						title="Similarity scores",
					)
					st.plotly_chart(fig_recs, use_container_width=True)
				except ValueError as exc:
					st.error(str(exc))
		else:
			album_type_options = sorted(df["album_type"].dropna().unique().tolist())
			col1, col2, col3 = st.columns(3)
			with col1:
				rec_album_type = st.selectbox("Preferred album type", album_type_options, key="rec_album_type")
				rec_bpm = st.slider("Preferred BPM", min_value=60, max_value=220, value=int(df["bpm"].median()), key="rec_bpm")
			with col2:
				rec_avg_ms = st.number_input(
					"Average ms played",
					min_value=10_000.0,
					max_value=600_000.0,
					value=float(df["average_ms_played"].median()),
					key="rec_avg_ms",
				)
				rec_avg_percent = st.slider(
					"Average % song played",
					min_value=30.0,
					max_value=100.0,
					value=float(df["avg_percent_song_played"].median()),
					key="rec_avg_percent",
				)
			with col3:
				rec_release_year = st.slider(
					"Preferred release year",
					min_value=int(df["release_year"].min()),
					max_value=2026,
					value=int(df["release_year"].median()),
					key="rec_release_year",
				)

			if st.button("Get recommendations", type="primary", key="profile_recs"):
				recs = rec_model.recommend_by_profile(
					album_type=rec_album_type,
					bpm=rec_bpm,
					avg_ms_played=rec_avg_ms,
					avg_percent=rec_avg_percent,
					release_year=rec_release_year,
					n=n_recs,
				)
				st.success("Tracks matching your listening profile")
				st.dataframe(recs, use_container_width=True, hide_index=True)

				fig_profile = px.scatter(
					recs,
					x="bpm",
					y="number_of_streams",
					size="score",
					color="album_type",
					hover_name="track_name",
					title="Recommended tracks by tempo and popularity",
				)
				st.plotly_chart(fig_profile, use_container_width=True)
