from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DEFAULT_NUM_FEATURES = [
    "release_year",
    "bpm",
    "average_ms_played",
    "avg_percent_song_played",
    "song_age_years",
    "average_played_sec",
    "bpm_x_completion",
]
DEFAULT_CAT_FEATURES = ["album_type"]
DISPLAY_COLS = [
    "track_name",
    "album_name",
    "album_type",
    "bpm",
    "number_of_streams",
    "avg_percent_song_played",
    "is_popular",
]


def build_preprocessor(
    num_features: list[str],
    cat_features: list[str],
) -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                num_features,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                cat_features,
            ),
        ],
        remainder="drop",
    )


class TrackRecommender:
    """Content-based track recommender using audio and listening features."""

    def __init__(
        self,
        n_neighbors: int = 6,
        popularity_weight: float = 0.15,
        reference_year: int = 2026,
    ) -> None:
        self.n_neighbors = n_neighbors
        self.popularity_weight = popularity_weight
        self.reference_year = reference_year
        self.num_features: list[str] = []
        self.cat_features: list[str] = []
        self.preprocessor: ColumnTransformer | None = None
        self.nn_model: NearestNeighbors | None = None
        self.catalog: pd.DataFrame | None = None
        self.feature_matrix: np.ndarray | None = None

    def _resolve_features(self, df: pd.DataFrame) -> tuple[list[str], list[str]]:
        num_features = [c for c in DEFAULT_NUM_FEATURES if c in df.columns]
        cat_features = [c for c in DEFAULT_CAT_FEATURES if c in df.columns]
        if not num_features:
            raise ValueError("No numeric recommendation features found in dataset.")
        return num_features, cat_features

    def _build_feature_row(
        self,
        album_type: str,
        bpm: float,
        avg_ms_played: float,
        avg_percent: float,
        release_year: int,
    ) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "release_year": release_year,
                    "bpm": bpm,
                    "average_ms_played": avg_ms_played,
                    "avg_percent_song_played": avg_percent,
                    "song_age_years": max(0, self.reference_year - release_year),
                    "average_played_sec": avg_ms_played / 1000.0,
                    "bpm_x_completion": bpm * avg_percent,
                    "album_type": album_type,
                }
            ]
        )

    def fit(self, df: pd.DataFrame) -> "TrackRecommender":
        catalog = df.dropna(subset=["track_name"]).copy().reset_index(drop=True)
        self.num_features, self.cat_features = self._resolve_features(catalog)

        feature_cols = self.num_features + self.cat_features
        self.preprocessor = build_preprocessor(self.num_features, self.cat_features)
        self.feature_matrix = self.preprocessor.fit_transform(catalog[feature_cols])

        self.nn_model = NearestNeighbors(
            n_neighbors=min(self.n_neighbors, len(catalog)),
            metric="cosine",
        )
        self.nn_model.fit(self.feature_matrix)
        self.catalog = catalog
        return self

    def _rank_candidates(
        self,
        distances: np.ndarray,
        indices: np.ndarray,
        exclude_indices: set[int] | None = None,
        n: int = 5,
    ) -> pd.DataFrame:
        if self.catalog is None:
            raise RuntimeError("Recommender is not fitted.")

        exclude_indices = exclude_indices or set()
        max_streams = max(self.catalog["number_of_streams"].max(), 1)
        rows: list[dict] = []

        for dist, idx in zip(distances, indices):
            idx = int(idx)
            if idx in exclude_indices:
                continue

            similarity = 1.0 - float(dist)
            popularity = self.catalog.loc[idx, "number_of_streams"] / max_streams
            score = similarity * (1.0 + self.popularity_weight * popularity)

            row = self.catalog.loc[idx, DISPLAY_COLS].to_dict()
            row["similarity"] = round(similarity, 4)
            row["score"] = round(score, 4)
            rows.append(row)

            if len(rows) >= n:
                break

        return pd.DataFrame(rows)

    def recommend_by_track(self, track_name: str, n: int = 5) -> pd.DataFrame:
        if self.catalog is None or self.nn_model is None:
            raise RuntimeError("Recommender is not fitted.")

        matches = self.catalog.index[self.catalog["track_name"] == track_name]
        if matches.empty:
            raise ValueError(f"Track not found: {track_name}")

        seed_idx = int(matches[0])
        distances, indices = self.nn_model.kneighbors(
            self.feature_matrix[seed_idx].reshape(1, -1),
            n_neighbors=min(self.n_neighbors, len(self.catalog)),
        )
        return self._rank_candidates(
            distances[0],
            indices[0],
            exclude_indices={seed_idx},
            n=n,
        )

    def recommend_by_profile(
        self,
        album_type: str,
        bpm: float,
        avg_ms_played: float,
        avg_percent: float,
        release_year: int,
        n: int = 5,
    ) -> pd.DataFrame:
        if self.preprocessor is None or self.nn_model is None:
            raise RuntimeError("Recommender is not fitted.")

        profile = self._build_feature_row(
            album_type=album_type,
            bpm=bpm,
            avg_ms_played=avg_ms_played,
            avg_percent=avg_percent,
            release_year=release_year,
        )
        profile_vector = self.preprocessor.transform(
            profile[self.num_features + self.cat_features]
        )
        distances, indices = self.nn_model.kneighbors(
            profile_vector,
            n_neighbors=min(self.n_neighbors, len(self.catalog)),
        )
        return self._rank_candidates(distances[0], indices[0], n=n)

    def recommend_similar_and_popular(
        self,
        track_name: str,
        n: int = 5,
    ) -> pd.DataFrame:
        """Blend nearest-neighbor similarity with catalog popularity."""
        similar = self.recommend_by_track(track_name, n=len(self.catalog))
        if similar.empty:
            return similar

        max_streams = max(similar["number_of_streams"].max(), 1)
        similar["blended_score"] = similar["similarity"] * (
            1.0 + self.popularity_weight * (similar["number_of_streams"] / max_streams)
        )
        return (
            similar.sort_values("blended_score", ascending=False)
            .head(n)
            .drop(columns=["blended_score"])
            .reset_index(drop=True)
        )


def train_recommender(
    data_path: Path,
    model_path: Path,
    n_neighbors: int = 6,
    popularity_weight: float = 0.15,
) -> TrackRecommender:
    df = pd.read_csv(data_path)
    recommender = TrackRecommender(
        n_neighbors=n_neighbors,
        popularity_weight=popularity_weight,
    ).fit(df)

    model_path.parent.mkdir(parents=True, exist_ok=True)
    import joblib

    joblib.dump(recommender, model_path)
    return recommender
