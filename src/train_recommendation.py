from pathlib import Path

from recommendation import train_recommender

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "src" / "feature_engineered_album.csv"
MODEL_PATH = ROOT / "models" / "track_recommender.pkl"


def main() -> None:
    recommender = train_recommender(
        data_path=DATA_PATH,
        model_path=MODEL_PATH,
        n_neighbors=6,
        popularity_weight=0.15,
    )

    sample_track = recommender.catalog.loc[0, "track_name"]
    sample_recs = recommender.recommend_by_track(sample_track, n=5)
    print(f"Saved recommender to {MODEL_PATH}")
    print(f"Catalog size: {len(recommender.catalog)} tracks")
    print(f"Sample recommendations for '{sample_track}':")
    print(sample_recs[["track_name", "album_name", "similarity", "score"]].to_string(index=False))


if __name__ == "__main__":
    main()
