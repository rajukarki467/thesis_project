"""
Reproduction package for Zeleke & Bochicchio (IEEE BigData 2024):
"Federated Kolmogorov-Arnold Networks for Health Data Analysis: A Study Using ECG Signal"
"""

from reproduction_zeleke2024.src.wavelet_preprocessing import (
    load_and_preprocess_dataset,
    apply_wavelet_denoising_db5,
    segment_and_extract_beats,
    partition_clients_dirichlet,
)
from reproduction_zeleke2024.src.model import (
    ZelekeKAN,
    KANLayer,
    MLPBaseline,
    build_model,
)
from reproduction_zeleke2024.src.client import (
    FederatedClient,
)
from reproduction_zeleke2024.src.server import (
    FederatedServer,
    evaluate_global_model,
)

__all__ = [
    "load_and_preprocess_dataset",
    "apply_wavelet_denoising_db5",
    "segment_and_extract_beats",
    "partition_clients_dirichlet",
    "ZelekeKAN",
    "KANLayer",
    "MLPBaseline",
    "build_model",
    "FederatedClient",
    "FederatedServer",
    "evaluate_global_model",
]
