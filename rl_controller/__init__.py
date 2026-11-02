"""
RL Controller Environment package for Resource-Adaptive Federated KANs.
"""

from rl_controller.env import (
    ACTION_TO_INDEX,
    ACTION_TUPLES,
    GRID_SIZES,
    NUM_ACTIONS_PER_CLIENT,
    SPLINE_ORDERS,
    FederatedKANEnv,
    RLControllerEnv,
    action_to_gk,
    compute_label_entropy,
    gk_to_action,
)

__all__ = [
    "FederatedKANEnv",
    "RLControllerEnv",
    "GRID_SIZES",
    "SPLINE_ORDERS",
    "ACTION_TUPLES",
    "ACTION_TO_INDEX",
    "NUM_ACTIONS_PER_CLIENT",
    "action_to_gk",
    "gk_to_action",
    "compute_label_entropy",
]
