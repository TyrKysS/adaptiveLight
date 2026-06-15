"""
Multi-goal Q-learning for closed-loop lux regulation.

Training happens entirely in simulation (digital twin from calibration curve).
State  : [lux_error_norm, brightness_norm, lux_trend_norm, prev_action_norm, goal_norm]
Q(s, g, a) — goal g is embedded as the 5th state dimension.
After simulation: epsilon=0, lr=lr_live so live operation only fine-tunes.
"""

import json
import os
import random
import threading
from collections import deque

import numpy as np

ACTIONS    = [-25, -15, -10, -5, -2, 0, 2, 5, 10, 15, 25]
N_ACTIONS  = len(ACTIONS)
STATE_SIZE = 5          # [err, br, trend, prev_action, goal]
MODEL_PATH = "/data/rl_model.json"
LUX_GOAL_SCALE = 1000.0  # goal normalisation constant


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


# ── Digital twin ──────────────────────────────────────────────────────────────

class DigitalTwin:
    """Simulated lux environment from calibration curve + Gaussian sensor noise."""

    def __init__(self, calib, noise_std: float = 2.0) -> None:
        self.calib     = calib
        self.noise_std = max(noise_std, 0.0)

    def step(self, brightness: float, delta: int) -> tuple:
        """Apply brightness delta; return (new_brightness, simulated_lux)."""
        new_br = float(np.clip(brightness + delta, 5, 100))
        lux    = self.calib.lux_for_brightness(new_br)
        if self.noise_std > 0:
            lux += float(np.random.normal(0.0, self.noise_std))
        return new_br, max(0.0, lux)


# ── RL agent ──────────────────────────────────────────────────────────────────

class RLAgent:
    """DQN multi-goal agent: online Q-net, frozen target net, experience replay."""

    LAYERS = [STATE_SIZE, 32, 16, N_ACTIONS]

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.q_net      = _NN(self.LAYERS)
        self.target_net = _NN(self.LAYERS)
        self.target_net.copy_weights_from(self.q_net)

        self.memory: deque  = deque(maxlen=4_000)
        self.epsilon:  float = 0.60
        self.epsilon_min: float = 0.05
        self.epsilon_decay: float = 0.99
        self.gamma:    float = 0.95
        self.lr:       float = 5e-4   # active learning rate; overwritten by set_live_mode
        self.lr_live:  float = 0.02   # fine-tuning LR after simulation
        self.batch:    int   = 8
        self.steps:    int   = 0
        self.target_freq: int = 50
        self.trained_by_sim: bool = False
        self._no_autosave:   bool = False

        # per-step tracking (updated by server.py during live operation)
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
        goal  = float(np.clip(target_lux / LUX_GOAL_SCALE, 0.0, 1.0))
        return np.array([[err, br, trend, pa, goal]], dtype=np.float32)

    @staticmethod
    def compute_reward(current_lux: float, target_lux: float,
                       prev_lux: "float | None", tolerance: float = 0.5) -> float:
        t   = max(target_lux, 1.0)
        err = abs(current_lux - t)
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

    def act_guarded(self, state: np.ndarray,
                    lux_val: float, target_lux: float) -> int:
        """
        Choose action with a directional safety guard for large lux errors.

        When lux deviates more than 50 % from target, the action is restricted
        to the correct direction so the network cannot move brightness the wrong
        way regardless of what Q-values it learned for out-of-distribution states
        (e.g. sensor covered → lux≈0 while brightness is still high).

        Inside the ±50 % band, behaviour is identical to act().
        """
        err_ratio = (lux_val - target_lux) / max(target_lux, 1.0)
        if err_ratio < -0.5:
            allowed = [i for i, a in enumerate(ACTIONS) if a >= 0]   # only increase
        elif err_ratio > 0.5:
            allowed = [i for i, a in enumerate(ACTIONS) if a <= 0]   # only decrease
        else:
            return self.act(state)   # normal ε-greedy, no restriction

        if random.random() < self.epsilon:
            return random.choice(allowed)
        with self._lock:
            qv = self.q_net.predict(state)[0]
        return int(max(allowed, key=lambda i: qv[i]))

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
        if not self._no_autosave and self.steps % 50 == 0:
            self.save()

    # ── simulation training ───────────────────────────────────────────────────

    def simulate(self, calib, goals: list, n_episodes: int = 1000,
                 steps_per_ep: int = 30, noise_std: float = 2.0,
                 progress_cb=None) -> None:
        """
        Offline training on a digital twin of the calibration curve.

        goals        : list of target lux values; each episode draws one at random.
        progress_cb  : optional callable(episode, total, epsilon) for status updates.

        On completion calls set_live_mode() and saves the model.
        """
        twin      = DigitalTwin(calib, noise_std)
        tolerance = 0.5
        lux_min   = max(1.0, min(p["lux"] for p in calib.points))
        lux_max   = max(lux_min + 1.0, calib.max_lux)

        self._no_autosave = True
        try:
            for ep in range(n_episodes):
                # Sample goal lux
                if goals:
                    goal_lux = float(random.choice(goals))
                else:
                    goal_lux = float(np.random.uniform(lux_min, lux_max))
                goal_lux = max(1.0, goal_lux)

                # 25 % of episodes start from a random brightness across the full range.
                # This teaches the agent what to do at extremes it might reach in real life
                # (e.g. brightness=5 % with lux still below target due to ambient light).
                if random.random() < 0.25:
                    init_br = float(np.random.uniform(5, 100))
                else:
                    init_br = calib.brightness_for_lux(goal_lux)
                    init_br = float(np.clip(init_br + np.random.normal(0, 5), 5, 100))
                cur_br, cur_lux = twin.step(init_br, 0)

                prev_lux = None
                prev_idx = ACTIONS.index(0)
                prev_s   = None

                for _ in range(steps_per_ep):
                    state      = self.build_state(cur_lux, goal_lux, cur_br,
                                                  prev_lux, prev_idx)
                    action_idx = self.act_guarded(state, cur_lux, goal_lux)
                    delta      = ACTIONS[action_idx]

                    new_br, new_lux = twin.step(cur_br, delta)

                    # 5 % of steps: inject a disturbance (sensor covered → lux≈0,
                    # or sudden bright ambient → lux≫max). Teaches the network the
                    # correct response for out-of-distribution sensor readings.
                    if random.random() < 0.05:
                        new_lux = 0.0 if random.random() < 0.5 else lux_max * 2.0

                    reward    = self.compute_reward(new_lux, goal_lux,
                                                    cur_lux, tolerance)
                    new_state = self.build_state(new_lux, goal_lux, new_br,
                                                 cur_lux, action_idx)

                    if prev_s is not None:
                        self.remember(prev_s, prev_idx, reward, state)
                        self.replay()

                    prev_s   = state
                    prev_idx = action_idx
                    prev_lux = cur_lux
                    cur_lux  = new_lux
                    cur_br   = new_br

                # Checkpoint and progress update every 100 episodes
                if ep % 100 == 0 or ep == n_episodes - 1:
                    self.save()
                    if progress_cb:
                        progress_cb(ep + 1, n_episodes, self.epsilon)

        finally:
            self._no_autosave = False

        self.set_live_mode()
        self.save()

    def set_live_mode(self) -> None:
        """Minimal exploration, very low LR — fine-tunes in real operation."""
        self.epsilon       = 0.05   # residual exploration prevents permanent stuck states
        self.epsilon_min   = 0.05
        self.lr            = self.lr_live
        self.trained_by_sim = True

    # ── persistence ───────────────────────────────────────────────────────────

    def save(self) -> None:
        try:
            os.makedirs("/data", exist_ok=True)
            with open(MODEL_PATH, "w") as fh:
                json.dump({
                    "q_net":          self.q_net.to_dict(),
                    "epsilon":        self.epsilon,
                    "steps":          self.steps,
                    "lr":             self.lr,
                    "trained_by_sim": self.trained_by_sim,
                }, fh)
        except Exception:
            pass

    def load(self) -> None:
        try:
            with open(MODEL_PATH) as fh:
                d = json.load(fh)
            net = _NN.from_dict(d["q_net"])
            if net.sizes[0] != STATE_SIZE:
                return  # incompatible checkpoint (e.g. old 4D model) — keep fresh init
            with self._lock:
                self.q_net = net
                self.target_net.copy_weights_from(self.q_net)
            self.epsilon        = float(d.get("epsilon", self.epsilon))
            self.steps          = int(d.get("steps", 0))
            self.lr             = float(d.get("lr", self.lr))
            self.trained_by_sim = bool(d.get("trained_by_sim", False))
        except Exception:
            pass

    def reset_model(self) -> None:
        with self._lock:
            self.q_net      = _NN(self.LAYERS)
            self.target_net = _NN(self.LAYERS)
            self.target_net.copy_weights_from(self.q_net)
        self.memory.clear()
        self.epsilon        = 0.60
        self.lr             = 5e-4
        self.trained_by_sim = False
        self.steps          = 0
        self.prev_state     = None
        self.prev_lux       = None
        self.last_reward    = None
        self.last_delta     = 0

    # ── info ──────────────────────────────────────────────────────────────────

    def info(self) -> dict:
        return {
            "epsilon":        round(self.epsilon, 4),
            "train_steps":    self.steps,
            "memory_size":    len(self.memory),
            "last_delta":     self.last_delta,
            "last_reward":    (round(self.last_reward, 3)
                               if self.last_reward is not None else None),
            "trained_by_sim": self.trained_by_sim,
        }
