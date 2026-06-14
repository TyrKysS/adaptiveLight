"""
Q-learning agent for closed-loop lux-based brightness regulation.

State  : [lux_error_norm, brightness_norm, lux_trend_norm, prev_action_norm]
Actions: discrete brightness-delta steps (percentage points, 0-100 scale)
Reward : proximity bonus — best at <5% error from target
"""

import json
import os
import random
import threading
from collections import deque

import numpy as np

ACTIONS    = [-25, -15, -10, -5, -2, 0, 2, 5, 10, 15, 25]
N_ACTIONS  = len(ACTIONS)
STATE_SIZE = 4
MODEL_PATH = "/data/rl_model.json"


# ── Neural network ────────────────────────────────────────────────────────────

class _NN:
    """3-layer feedforward net; He init, ReLU hidden, linear output, SGD backprop."""

    def __init__(self, sizes: list):
        self.sizes = sizes
        rng = np.random.default_rng(0)
        self.W = [rng.standard_normal((a, b)) * np.sqrt(2.0 / a)
                  for a, b in zip(sizes, sizes[1:])]
        self.b = [np.zeros((1, s)) for s in sizes[1:]]

    def predict(self, x: np.ndarray) -> np.ndarray:
        for i, (w, b) in enumerate(zip(self.W, self.b)):
            x = x @ w + b
            if i < len(self.W) - 1:
                x = np.maximum(0.0, x)
        return x

    def sgd_step(self, x: np.ndarray, y_target: np.ndarray, lr: float) -> None:
        acts = [x]
        for i, (w, b) in enumerate(zip(self.W, self.b)):
            z = acts[-1] @ w + b
            acts.append(np.maximum(0.0, z) if i < len(self.W) - 1 else z)
        delta = acts[-1] - y_target
        for i in range(len(self.W) - 1, -1, -1):
            self.W[i] -= lr * acts[i].T @ delta
            self.b[i] -= lr * delta
            if i > 0:
                delta = (delta @ self.W[i].T) * (acts[i] > 0)

    def copy_weights_from(self, src: "_NN") -> None:
        self.W = [w.copy() for w in src.W]
        self.b = [b.copy() for b in src.b]

    def to_dict(self) -> dict:
        return {"sizes": self.sizes,
                "W": [w.tolist() for w in self.W],
                "b": [b.tolist() for b in self.b]}

    @classmethod
    def from_dict(cls, d: dict) -> "_NN":
        inst = cls.__new__(cls)
        inst.sizes = d["sizes"]
        inst.W = [np.array(w) for w in d["W"]]
        inst.b = [np.array(b) for b in d["b"]]
        return inst


# ── RL agent ──────────────────────────────────────────────────────────────────

class RLAgent:
    """DQN-style agent: online Q-net, frozen target net, experience replay."""

    LAYERS = [STATE_SIZE, 32, 16, N_ACTIONS]

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.q_net     = _NN(self.LAYERS)
        self.target_net = _NN(self.LAYERS)
        self.target_net.copy_weights_from(self.q_net)

        self.memory: deque = deque(maxlen=4_000)
        self.epsilon:       float = 0.60
        self.epsilon_min:   float = 0.05
        self.epsilon_decay: float = 0.99
        self.gamma:         float = 0.95
        self.lr:            float = 5e-4
        self.batch:         int   = 8
        self.steps:         int   = 0
        self.target_freq:   int   = 50

        # per-step tracking (updated by server.py)
        self.prev_state:      "np.ndarray | None" = None
        self.prev_action_idx: int   = ACTIONS.index(0)
        self.prev_lux:        "float | None" = None
        self.last_reward:     "float | None" = None
        self.last_delta:      int   = 0

        self.load()

    # ── state / reward ────────────────────────────────────────────────────────

    @staticmethod
    def build_state(current_lux: float, target_lux: float,
                    brightness: float, prev_lux: "float | None",
                    prev_action_idx: int) -> np.ndarray:
        t     = max(target_lux, 1.0)
        err   = float(np.clip((current_lux - t) / t, -3.0, 3.0))
        br    = float(np.clip(brightness / 100.0, 0.0, 1.0))
        trend = (0.0 if prev_lux is None
                 else float(np.clip((current_lux - prev_lux) / t, -1.0, 1.0)))
        pa    = ACTIONS[prev_action_idx] / 25.0
        return np.array([[err, br, trend, pa]], dtype=np.float32)

    @staticmethod
    def compute_reward(current_lux: float, target_lux: float,
                       prev_lux: "float | None", tolerance: float = 0.5) -> float:
        t   = max(target_lux, 1.0)
        err = abs(current_lux - t)
        # Dead band: within ±tolerance lux is considered "at target"
        if err <= tolerance:
            return 2.0
        pct = err / t
        r   = -pct * 2.0
        if pct < 0.15:
            r += 0.5
        if prev_lux is not None and err < abs(prev_lux - t):
            r += 0.4
        return float(np.clip(r, -3.0, 3.0))

    # ── policy ────────────────────────────────────────────────────────────────

    def act(self, state: np.ndarray) -> int:
        if random.random() < self.epsilon:
            return random.randrange(N_ACTIONS)
        with self._lock:
            return int(np.argmax(self.q_net.predict(state)))

    # ── training ──────────────────────────────────────────────────────────────

    def remember(self, s: np.ndarray, a: int, r: float, s2: np.ndarray) -> None:
        self.memory.append((s, a, r, s2))

    def replay(self) -> None:
        if len(self.memory) < self.batch:
            return
        batch = random.sample(self.memory, self.batch)
        with self._lock:
            for s, a, r, s2 in batch:
                tq = self.q_net.predict(s).copy()
                tq[0, a] = r + self.gamma * float(np.max(self.target_net.predict(s2)))
                self.q_net.sgd_step(s, tq, self.lr)
        self.steps    += 1
        self.epsilon   = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        if self.steps % self.target_freq == 0:
            with self._lock:
                self.target_net.copy_weights_from(self.q_net)
        if self.steps % 50 == 0:
            self.save()

    # ── persistence ───────────────────────────────────────────────────────────

    def save(self) -> None:
        try:
            os.makedirs("/data", exist_ok=True)
            with open(MODEL_PATH, "w") as fh:
                json.dump({"q_net": self.q_net.to_dict(),
                           "epsilon": self.epsilon,
                           "steps": self.steps}, fh)
        except Exception:
            pass

    def load(self) -> None:
        try:
            with open(MODEL_PATH) as fh:
                d = json.load(fh)
            with self._lock:
                self.q_net = _NN.from_dict(d["q_net"])
                self.target_net.copy_weights_from(self.q_net)
            self.epsilon = float(d.get("epsilon", self.epsilon))
            self.steps   = int(d.get("steps", 0))
        except Exception:
            pass

    def reset_model(self) -> None:
        with self._lock:
            self.q_net      = _NN(self.LAYERS)
            self.target_net = _NN(self.LAYERS)
            self.target_net.copy_weights_from(self.q_net)
        self.memory.clear()
        self.epsilon     = 0.60
        self.steps       = 0
        self.prev_state  = None
        self.prev_lux    = None
        self.last_reward = None
        self.last_delta  = 0

    # ── info ──────────────────────────────────────────────────────────────────

    def info(self) -> dict:
        return {
            "epsilon":     round(self.epsilon, 4),
            "train_steps": self.steps,
            "memory_size": len(self.memory),
            "last_delta":  self.last_delta,
            "last_reward": (round(self.last_reward, 3)
                            if self.last_reward is not None else None),
        }
