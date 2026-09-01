from pathlib import Path
from typing import List, Optional

import wfdb


def get_record_ids(raw_dir: Path) -> List[str]:
    records_file = raw_dir / "RECORDS"

    if not records_file.exists():
        raise FileNotFoundError(
            f"RECORDS file does not exist: {records_file}"
        )

    return [
        line.strip()
        for line in records_file.read_text().splitlines()
        if line.strip()
    ]


def load_record(raw_dir: Path, record_id: str):
    record = wfdb.rdrecord(
        str(raw_dir / record_id)
    )

    annotation = wfdb.rdann(
        str(raw_dir / record_id),
        "atr"
    )

    return record, annotation