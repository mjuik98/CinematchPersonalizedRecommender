from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pathlib import Path

from app.config import settings
from app.data import create_synthetic_sample_dataset

if __name__ == "__main__":
    dataset = create_synthetic_sample_dataset(settings.sample_data_dir)
    print(f"Created synthetic dataset at {settings.sample_data_dir} with {len(dataset.users)} users, {len(dataset.items)} items, {len(dataset.interactions)} interactions.")
