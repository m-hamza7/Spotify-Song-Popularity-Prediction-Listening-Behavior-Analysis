from pathlib import Path
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[1] / 'data'

def load_all():
    files = {
        'album': DATA_DIR / 'spotify_skz_counts_album.csv',
        'day': DATA_DIR / 'spotify_skz_counts_day.csv',
        'streaming': DATA_DIR / 'spotify_skz_streaming_4.25_4.26.csv'
    }
    dfs = {}
    for k, p in files.items():
        if p.exists():
            dfs[k] = pd.read_csv(p)
        else:
            dfs[k] = None
    return dfs

if __name__ == '__main__':
    data = load_all()
    for k, v in data.items():
        if v is None:
            print(f"{k}: file not found")
        else:
            print(f"{k}: loaded {len(v)} rows")
