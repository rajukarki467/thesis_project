"""
Proximal Policy Optimization (PPO) Controller for Resource-Adaptive Federated KAN.

Mathematical Formulation (Section 3.2.4 & Section 3.2.5):
Uses an Actor-Critic architecture with clipped surrogate objective (Equation 8):
    L^CLIP(θ) = E_t [ min( r_t(θ) * A_t, clip(r_t(θ), 1-ε, 1+ε) * A_t ) ]
where r_t(θ) = π_θ(a_t | s_t) / π_{θ_old}(a_t | s_t) and A_t is the GAE generalized advantage.

Interacts with the Gymnasium FederatedKANEnv:
- State Space: Box(low=0.0, high=1.0, shape=(N, 5), dtype=np.float32)
- Action Space: MultiDiscrete([12] * N) for (G_i, k_i) in {3, 5, 7, 10} x {2, 3, 4}
"""

from __future__ import annotations

import csv
import logging
import os
from pathlib import Path
import sys
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import gymnasium as gym
import numpy as np
import stable_baselines3 as sb3
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
import torch
import torch.nn as nn
import yaml

logger = logging.getLogger("ppo_controller")


class PPOLoggingCallback(BaseCallback):
    """
    Custom callback to log policy loss, value loss, entropy, and rolling episode reward
    to CSV files in logs/ppo/ for monitoring training convergence.
    """

    def __init__(
        self,
        log_dir: Union[str, Path] = "logs/ppo",
        log_filename: str = "ppo_training_log.csv",
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.log_dir / log_filename

        self.episode_count: int = 0
        self.episode_rewards: List[float] = []
        self.episode_lengths: List[int] = []
        self.current_ep_reward: float = 0.0
        self.current_ep_length: int = 0

        # Initialize CSV header if file does not exist
        if not self.log_file.exists():
            with open(self.log_file, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "step",
                    "episode",
                    "episode_reward",
                    "rolling_mean_reward",
                    "episode_length",
                    "policy_loss",
                    "value_loss",
                    "entropy_loss",
                    "approx_kl",
                    "clip_fraction",
                    "explained_variance",
                ])

    def _on_step(self) -> bool:
        """Invoked after each step in the environment."""
        # Accumulate rewards for active episodes
        rewards = self.locals.get("rewards", [0.0])
        dones = self.locals.get("dones", [False])

        self.current_ep_reward += float(rewards[0]) if len(rewards) > 0 else 0.0
        self.current_ep_length += 1

        if len(dones) > 0 and dones[0]:
            self.episode_count += 1
            self.episode_rewards.append(self.current_ep_reward)
            self.episode_lengths.append(self.current_ep_length)

            rolling_mean = float(np.mean(self.episode_rewards[-10:]))

            # Extract SB3 training metrics if available
            logger_dict = getattr(self.model, "logger", None)
            name_to_val = getattr(logger_dict, "name_to_value", {}) if logger_dict else {}

            p_loss = name_to_val.get("train/policy_gradient_loss", "")
            v_loss = name_to_val.get("train/value_loss", "")
            ent = name_to_val.get("train/entropy_loss", "")
            kl = name_to_val.get("train/approx_kl", "")
            clip_frac = name_to_val.get("train/clip_fraction", "")
            expl_var = name_to_val.get("train/explained_variance", "")

            with open(self.log_file, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    self.num_timesteps,
                    self.episode_count,
                    f"{self.current_ep_reward:.5f}",
                    f"{rolling_mean:.5f}",
                    self.current_ep_length,
                    f"{p_loss:.6f}" if isinstance(p_loss, (int, float)) else "",
                    f"{v_loss:.6f}" if isinstance(v_loss, (int, float)) else "",
                    f"{ent:.6f}" if isinstance(ent, (int, float)) else "",
                    f"{kl:.6f}" if isinstance(kl, (int, float)) else "",
                    f"{clip_frac:.6f}" if isinstance(clip_frac, (int, float)) else "",
                    f"{expl_var:.6f}" if isinstance(expl_var, (int, float)) else "",
                ])

            # Reset episode accumulators
            self.current_ep_reward = 0.0
            self.current_ep_length = 0

        return True


class PPOController:
    """
    PPO Meta-Controller for Resource-Adaptive Federated KAN.

    Wraps Stable-Baselines3 PPO with configuration loading from configs/ppo.yaml,
    factorized actor-critic networks for MultiDiscrete actions, and evaluation helpers.

    Parameters
    ----------
    env : Optional[gym.Env]
        Gymnasium environment instance.
    config_path : Optional[str | Path]
        Path to PPO YAML configuration.
    device : Optional[str | torch.device]
        Target execution device ('cpu' or 'cuda').
    verbose : int
        Verbosity level (default 1).
    """

    def __init__(
        self,
        env: Optional[gym.Env] = None,
        config_path: Optional[Union[str, Path]] = "configs/ppo.yaml",
        device: Optional[Union[str, torch.device]] = None,
        verbose: int = 1,
        **kwargs,
    ):
        self.config = self._load_config(config_path)
        ppo_cfg = self.config.get("ppo", {})

        self.env = env
        self.verbose = verbose

        # Determine device
        if device is not None:
            self.device_str = str(device)
        else:
            self.device_str = "cuda" if torch.cuda.is_available() else "cpu"

        # Extract PPO hyperparameters with kwargs override support
        self.learning_rate = float(kwargs.pop("learning_rate", ppo_cfg.get("learning_rate", 3e-4)))
        self.gamma = float(kwargs.pop("gamma", ppo_cfg.get("gamma", 0.99)))
        self.gae_lambda = float(kwargs.pop("gae_lambda", ppo_cfg.get("gae_lambda", 0.95)))
        self.clip_range = float(kwargs.pop("clip_range", ppo_cfg.get("clip_range", 0.2)))
        self.clip_range_vf = kwargs.pop("clip_range_vf", ppo_cfg.get("clip_range_vf", None))
        self.ent_coef = float(kwargs.pop("ent_coef", ppo_cfg.get("ent_coef", 0.01)))
        self.vf_coef = float(kwargs.pop("vf_coef", ppo_cfg.get("vf_coef", 0.5)))
        self.max_grad_norm = float(kwargs.pop("max_grad_norm", ppo_cfg.get("max_grad_norm", 0.5)))
        self.n_steps = int(kwargs.pop("n_steps", ppo_cfg.get("n_steps", 64)))
        self.batch_size = int(kwargs.pop("batch_size", ppo_cfg.get("batch_size", 32)))
        self.n_epochs = int(kwargs.pop("n_epochs", ppo_cfg.get("n_epochs", 10)))
        self.seed = int(kwargs.pop("seed", self.config.get("environment", {}).get("seed", 42)))

        # Actor-Critic Architecture (shared feature extractor + separate heads)
        policy_kwargs = kwargs.pop("policy_kwargs", dict(
            net_arch=dict(pi=[128, 128], vf=[128, 128]),
            activation_fn=nn.ReLU,
        ))

        if self.env is not None:
            self.model = PPO(
                policy="MlpPolicy",
                env=self.env,
                learning_rate=self.learning_rate,
                n_steps=self.n_steps,
                batch_size=self.batch_size,
                n_epochs=self.n_epochs,
                gamma=self.gamma,
                gae_lambda=self.gae_lambda,
                clip_range=self.clip_range,
                clip_range_vf=self.clip_range_vf,
                ent_coef=self.ent_coef,
                vf_coef=self.vf_coef,
                max_grad_norm=self.max_grad_norm,
                policy_kwargs=policy_kwargs,
                verbose=self.verbose,
                seed=self.seed,
                device=self.device_str,
                **kwargs,
            )
            self._init_policy_bias()
        else:
            self.model = None

    def _init_policy_bias(self) -> None:
        """
        Initialize action_net bias towards baseline (action 4: G=5, k=3) and
        efficient configurations (actions 0..3) to prevent initial over-parameterization.
        """
        if self.model is None or not hasattr(self.model, "policy"):
            return
        action_net = getattr(self.model.policy, "action_net", None)
        if action_net is not None and getattr(action_net, "bias", None) is not None:
            with torch.no_grad():
                b = action_net.bias
                num_logits = b.numel()
                num_clients = num_logits // 12
                for c in range(num_clients):
                    # Actions 0..3: +0.5 (efficient configurations)
                    b[c * 12 : c * 12 + 4] = 0.5
                    # Action 4: +1.5 (FedAvg-KAN baseline G=5, k=3)
                    b[c * 12 + 4] = 1.5
                    # Actions 5..11: -1.5 (over-parameterized configurations)
                    b[c * 12 + 5 : (c + 1) * 12] = -1.5

    def _load_config(self, config_path: Optional[Union[str, Path]]) -> Dict[str, Any]:
        """Load YAML configuration or return sensible defaults."""
        if config_path is not None:
            path = Path(config_path)
            if not path.is_absolute():
                path = PROJECT_ROOT / path
            if path.exists():
                try:
                    with open(path, "r") as f:
                        return yaml.safe_load(f) or {}
                except Exception as e:
                    logger.warning(f"Could not load config from {path}: {e}")
        return {}

    def set_env(self, env: gym.Env) -> None:
        """Bind environment to PPO controller."""
        self.env = env
        if self.model is not None:
            self.model.set_env(env)
        else:
            policy_kwargs = dict(
                net_arch=dict(pi=[128, 128], vf=[128, 128]),
                activation_fn=nn.ReLU,
            )
            self.model = PPO(
                policy="MlpPolicy",
                env=self.env,
                learning_rate=self.learning_rate,
                n_steps=self.n_steps,
                batch_size=self.batch_size,
                n_epochs=self.n_epochs,
                gamma=self.gamma,
                gae_lambda=self.gae_lambda,
                clip_range=self.clip_range,
                clip_range_vf=self.clip_range_vf,
                ent_coef=self.ent_coef,
                vf_coef=self.vf_coef,
                max_grad_norm=self.max_grad_norm,
                policy_kwargs=policy_kwargs,
                verbose=self.verbose,
                seed=self.seed,
                device=self.device_str,
            )
            self._init_policy_bias()

    def train(
        self,
        total_timesteps: int,
        callback: Optional[BaseCallback] = None,
        log_dir: str = "logs/ppo",
    ) -> PPOController:
        """
        Train the PPO meta-controller on the federated environment.

        Parameters
        ----------
        total_timesteps : int
            Total environment interaction steps to execute.
        callback : Optional[BaseCallback]
            Custom callback instance.
        log_dir : str
            Directory to store training logs.

        Returns
        -------
        PPOController
            Trained controller instance.
        """
        if self.model is None or self.env is None:
            raise ValueError("Environment must be set before training. Call set_env(env).")

        # Setup logging callback
        if callback is None:
            callback = PPOLoggingCallback(log_dir=log_dir)

        logger.info(f"Starting PPO training for {total_timesteps} timesteps on {self.device_str}...")
        self.model.learn(total_timesteps=total_timesteps, callback=callback)
        logger.info("PPO training completed successfully.")
        return self

    def predict(
        self,
        observation: np.ndarray,
        deterministic: bool = True,
    ) -> np.ndarray:
        """
        Predict (G_i, k_i) action configurations for all clients given observation S^t.

        Parameters
        ----------
        observation : np.ndarray
            State matrix of shape (N, 5).
        deterministic : bool
            Whether to use deterministic argmax actions (default True).

        Returns
        -------
        np.ndarray
            1D array of shape (N,) with action indices in [0..11].
        """
        if self.model is None:
            raise ValueError("Model is not initialized. Train or load weights first.")

        action, _ = self.model.predict(observation, deterministic=deterministic)
        return np.asarray(action, dtype=np.int64).flatten()

    def save(self, path: Union[str, Path]) -> None:
        """Serialize PPO policy weights and optimizer state to disk."""
        if self.model is None:
            raise ValueError("Cannot save uninitialized model.")
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save(str(save_path))
        logger.info(f"Saved PPO controller checkpoint to {save_path}")

    def load(
        self,
        path: Union[str, Path],
        env: Optional[gym.Env] = None,
    ) -> PPOController:
        """Load saved PPO controller checkpoint into this instance."""
        target_env = env if env is not None else self.env
        self.model = PPO.load(str(path), env=target_env, device=self.device_str)
        if target_env is not None:
            self.env = target_env
        logger.info(f"Loaded PPO controller checkpoint from {path}")
        return self

    @classmethod
    def load_from_checkpoint(
        cls,
        path: Union[str, Path],
        env: Optional[gym.Env] = None,
        config_path: str = "configs/ppo.yaml",
    ) -> PPOController:
        """Factory classmethod to construct and load PPOController."""
        inst = cls(config_path=config_path, env=env, verbose=0)
        inst.load(path, env=env)
        return inst
