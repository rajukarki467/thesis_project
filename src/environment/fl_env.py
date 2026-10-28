"""
Alias module for FederatedKANEnv (re-exporting from f1_env).
"""

from src.environment.f1_env import (
    ACTION_TO_INDEX,
    ACTION_TUPLES,
    GRID_SIZES,
    NUM_ACTIONS_PER_CLIENT,
    SPLINE_ORDERS,
    FederatedKANEnv,
    action_to_gk,
    gk_to_action,
    run_sanity_check,
)

__all__ = [
    "FederatedKANEnv",
    "GRID_SIZES",
    "SPLINE_ORDERS",
    "ACTION_TUPLES",
    "ACTION_TO_INDEX",
    "NUM_ACTIONS_PER_CLIENT",
    "action_to_gk",
    "gk_to_action",
    "run_sanity_check",
]
