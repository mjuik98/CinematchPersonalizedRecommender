from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pathlib import Path
import shutil

from app.config import settings

if __name__ == "__main__":
    latest_json = settings.latest_report_path
    if not latest_json.exists():
        raise SystemExit("No latest report found. Run training first.")
    latest = __import__("json").loads(latest_json.read_text(encoding="utf-8"))
    html_path = Path(latest["html_path"])
    output_path = settings.project_root / "docs" / "sample_eval_report.html"
    shutil.copyfile(html_path, output_path)
    print(f"Copied {html_path} -> {output_path}")
