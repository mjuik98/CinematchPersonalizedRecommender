from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
import base64
import io
import json

import matplotlib.pyplot as plt
import pandas as pd
from jinja2 import Environment, FileSystemLoader, select_autoescape


def _fig_to_base64(fig) -> str:
    buffer = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buffer, format="png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def create_metric_chart(metrics_df: pd.DataFrame, columns: list[str], title: str) -> str:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    metrics_df.set_index("model")[columns].plot(kind="bar", ax=ax)
    ax.set_title(title)
    ax.set_ylabel("score")
    ax.set_ylim(0, max(1.0, metrics_df[columns].to_numpy().max() * 1.15))
    ax.legend(loc="best")
    return _fig_to_base64(fig)


def render_html_report(
    templates_dir: Path,
    report_payload: dict[str, Any],
    output_html_path: Path,
) -> None:
    env = Environment(
        loader=FileSystemLoader(str(templates_dir)),
        autoescape=select_autoescape(enabled_extensions=("html",)),
    )
    template = env.get_template("report_template.html")
    html = template.render(**report_payload)
    output_html_path.parent.mkdir(parents=True, exist_ok=True)
    output_html_path.write_text(html, encoding="utf-8")


def save_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
