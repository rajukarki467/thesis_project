from pathlib import Path
from typing import List, Optional, Tuple, Union
import wfdb

DEFAULT_RAW_DIR = Path("data/raw/mit-bih-arrhythmia-database-1.0.0")


def get_record_ids(raw_dir: Optional[Union[str, Path]] = None) -> List[str]:
    """
    Return all 48 MIT-BIH record identifiers from the RECORDS file.
    """
    if raw_dir is None:
        raw_dir = DEFAULT_RAW_DIR
    raw_dir = Path(raw_dir)
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


# Alias for backward compatibility
get_record_names = get_record_ids


def _resolve_record_args(
    arg1: Union[str, Path],
    arg2: Optional[Union[str, Path]] = None,
    raw_dir: Optional[Union[str, Path]] = None,
) -> Tuple[Path, str]:
    """
    Helper to resolve (raw_dir, record_id) regardless of whether called as:
      - load_record(record_id)
      - load_record(raw_dir, record_id)
      - load_record(record_id, raw_dir)
    """
    if arg2 is None:
        if raw_dir is None:
            raw_path = DEFAULT_RAW_DIR
        else:
            raw_path = Path(raw_dir)
        record_id = str(arg1)
    else:
        # Determine which is directory and which is record_id
        p1 = Path(arg1)
        p2 = Path(arg2)
        if (p1 / f"{p2}.hea").exists() or (p1 / f"{p2}.dat").exists():
            raw_path, record_id = p1, str(arg2)
        elif (p2 / f"{p1}.hea").exists() or (p2 / f"{p1}.dat").exists():
            raw_path, record_id = p2, str(arg1)
        elif p1.is_dir():
            raw_path, record_id = p1, str(arg2)
        else:
            raw_path, record_id = p2, str(arg1)

    return raw_path, record_id


def load_record(
    arg1: Union[str, Path],
    arg2: Optional[Union[str, Path]] = None,
    raw_dir: Optional[Union[str, Path]] = None,
):
    """
    Load a WFDB record from the dataset directory.
    """
    raw_path, record_id = _resolve_record_args(arg1, arg2, raw_dir)
    return wfdb.rdrecord(str(raw_path / record_id))


def load_annotations(
    arg1: Union[str, Path],
    arg2: Optional[Union[str, Path]] = None,
    raw_dir: Optional[Union[str, Path]] = None,
):
    """
    Load WFDB annotations (.atr) from the dataset directory.
    """
    raw_path, record_id = _resolve_record_args(arg1, arg2, raw_dir)
    return wfdb.rdann(str(raw_path / record_id), "atr")
