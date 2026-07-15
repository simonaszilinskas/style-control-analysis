"""Shared repo paths, resolved relative to the repo root so scripts run from anywhere."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"

for _d in (DATA, RESULTS, FIGURES):
    _d.mkdir(exist_ok=True)

BATTLES = DATA / "fr_battles.parquet"
