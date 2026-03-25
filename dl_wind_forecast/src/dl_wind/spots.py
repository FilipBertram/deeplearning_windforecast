from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path

@dataclass
class SpotConfig:
    name: str
    dataset_path: Path
    splits_path: Path


def get_spot_configs(project_root: Path) -> dict[str, SpotConfig]:
    base = project_root / "data/raw/windguru_data"

    return {
        "Holnis": SpotConfig(
            name="Holnis",
            dataset_path=base / "aide/station3402_Holnis/ml_ready_12h_pipeline/final_12h.parquet",
            splits_path=project_root / "data/processed/splits_holnis_12h/splits.json",
        ),
        "Schausende": SpotConfig(
            name="Schausende",
            dataset_path=base / "aide/station3529_schausende/ml_ready_12h_pipeline/final_12h.parquet",
            splits_path=project_root / "data/processed/splits_schausende_12h/splits.json",
        ),
        "Kollund": SpotConfig(
            name="Kollund",
            dataset_path=base / "aide/station3846_kollund/ml_ready_12h_pipeline/final_12h.parquet",
            splits_path=project_root / "data/processed/splits_kollund_12h/splits.json",
        ),
        "Wackerballig": SpotConfig(
            name="Wackerballig",
            dataset_path=base / "aide/station3737_wackerballig/ml_ready_12h_pipeline/final_12h.parquet",
            splits_path=project_root / "data/processed/splits_wackerballig_12h/splits.json",
        ),
    }