from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import settings
from app.data import download_movielens_1m

if __name__ == "__main__":
    path = download_movielens_1m(settings.raw_dir)
    print(f"Downloaded MovieLens 1M to {path}")
