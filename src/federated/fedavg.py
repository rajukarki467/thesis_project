"""
FedAvg (Federated Averaging) Utilities and Strategy.
"""

from flwr.server.strategy import FedAvg
from src.federated.server import aggregate_parameters

__all__ = ["FedAvg", "aggregate_parameters"]
