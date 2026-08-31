from pathlib import Path 
from typing import List,Dict,Optional

import numpy as np
import pandas as pd 
import wfdb



PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_DATA_DIR = (
    PROJECT_ROOT /"data" /"raw" /"mit-bih-arrhythmia-database-1.0.0"

)

def get_record_name(data_dir:Optional[Path] =None) -> List[str] :
    """
    Discover all WFDB record from .hea files.
    
    Parameters
    ----------
    data_dir : str,Optional
    Directory containing MIT-BIH Arrhythmia Database .hea/.dat/.atr files.

    Returns
    ----------
    list[str]
    Sorted record names without file extensions.

    """

    data_path = Path(data_dir) if data_dir else DEFAULT_DATA_DIR
    if not data_path.exists():
        raise FileNotFoundError(F"Dataset directory doesnot exist: \n{data_path}")

    records = sorted(
        file.stem for file in data_path.glob("*.hea")
        
    )