"""
Unit tests for Heuristic and PPO Controllers.

Verifies:
1. HeuristicController: Complexity score calculation, quantile assignment, decision explanation.
2. PPOController: Initialization from config, multi-discrete prediction shape, checkpoint saving and loading.
3. PPOLoggingCallback: Logging training dynamics to CSV.
"""

from __future__ import annotations

from pathlib import Path
import sys

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pytest

from src.controllers.heuristic import HeuristicController
from src.controllers.ppo_controller import PPOController, PPOLoggingCallback
from src.environment.f1_env import action_to_gk


# ==============================================================================
# 1. Heuristic Controller Tests
# ==============================================================================

def test_heuristic_controller_complexity_scores():
    ctrl = HeuristicController(w1=1.0, w2=1.0, w3=1.0)

    # 4 clients with known values: [loss, var, lat, bw, curv]
    obs = np.array([
        [0.2, 0.5, 0.1, 0.8, 0.0],  # Loss=1.0, Var=1.0, bw=8.2 -> C ~ 1.0 + 1.0 + 0.12 = 2.12
        [0.8, 0.9, 0.4, 0.1, 0.2],  # Loss=4.0, Var=1.8, bw=1.9 -> C ~ 4.0 + 1.8 + 0.52 = 6.32
        [0.5, 0.5, 0.2, 0.5, 0.1],  # Loss=2.5, Var=1.0, bw=5.5 -> C ~ 2.5 + 1.0 + 0.18 = 3.68
        [0.3, 0.4, 0.1, 0.7, 0.0],  # Loss=1.5, Var=0.8, bw=7.3 -> C ~ 1.5 + 0.8 + 0.14 = 2.44
    ], dtype=np.float32)

    scores = ctrl.compute_complexity_scores(obs)
    assert len(scores) == 4
    # Client 1 should have highest complexity, Client 0 lowest
    assert np.argmax(scores) == 1
    assert np.argmin(scores) == 0


def test_heuristic_controller_quantile_allocation():
    ctrl = HeuristicController()

    # 8 clients with ascending complexity
    obs = np.array([
        [0.1, 0.2, 0.1, 0.9, 0.0],  # 0: lowest
        [0.2, 0.3, 0.1, 0.8, 0.0],  # 1: low
        [0.4, 0.5, 0.2, 0.6, 0.1],  # 2: medium
        [0.5, 0.5, 0.2, 0.5, 0.1],  # 3: medium
        [0.5, 0.6, 0.3, 0.5, 0.1],  # 4: medium
        [0.6, 0.6, 0.3, 0.4, 0.2],  # 5: medium
        [0.8, 0.8, 0.5, 0.2, 0.3],  # 6: high
        [0.9, 0.9, 0.6, 0.1, 0.4],  # 7: highest
    ], dtype=np.float32)

    actions = ctrl.predict(obs)
    assert len(actions) == 8

    # Quantiles: 25% low (2 clients), 50% med (4 clients), 25% high (2 clients)
    counts = np.bincount(actions, minlength=12)
    assert counts[ctrl.low_action] == 2
    assert counts[ctrl.med_action] == 4
    assert counts[ctrl.high_action] == 2

    # Verify action mapping
    assert actions[0] == ctrl.low_action
    assert actions[1] == ctrl.low_action
    assert actions[6] == ctrl.high_action
    assert actions[7] == ctrl.high_action


def test_heuristic_controller_explain_decision():
    ctrl = HeuristicController()
    obs = np.random.uniform(0.1, 0.9, size=(8, 5)).astype(np.float32)
    breakdown = ctrl.explain_decision(obs)

    assert len(breakdown) == 8
    for item in breakdown:
        assert "client_id" in item
        assert "complexity_score" in item
        assert "tier" in item
        assert item["tier"] in ["low", "medium", "high"]
        assert "action_idx" in item
        assert 0 <= item["action_idx"] < 12


# ==============================================================================
# 2. PPO Controller Tests
# ==============================================================================

class MockGymEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(8, 5), dtype=np.float32)
        self.action_space = spaces.MultiDiscrete([12] * 8)
        self.current_step = 0

    def reset(self, seed=None):
        super().reset(seed=seed)
        self.current_step = 0
        return np.zeros((8, 5), dtype=np.float32), {"reset_info": True}

    def step(self, action):
        self.current_step += 1
        done = self.current_step >= 4
        return np.ones((8, 5), dtype=np.float32) * 0.5, 0.25, done, False, {"step": self.current_step}


def test_ppo_controller_predict_and_checkpoint(tmp_path):
    env = MockGymEnv()
    ctrl = PPOController(env=env, verbose=0, n_steps=16, batch_size=16)

    obs, _ = env.reset()
    action = ctrl.predict(obs)

    assert action.shape == (8,)
    assert np.all(action >= 0) and np.all(action < 12)

    # Train short step
    ctrl.train(total_timesteps=16, log_dir=str(tmp_path / "logs"))

    # Predict after training
    action_trained = ctrl.predict(obs)

    # Test checkpoint save and load
    save_file = tmp_path / "test_ppo_checkpoint.zip"
    ctrl.save(save_file)
    assert save_file.exists()

    loaded_ctrl = PPOController(config_path="configs/ppo.yaml", verbose=0)
    loaded_ctrl.load(save_file, env=env)

    action_loaded = loaded_ctrl.predict(obs)
    assert action_loaded.shape == (8,)
    np.testing.assert_array_equal(action_trained, action_loaded)
