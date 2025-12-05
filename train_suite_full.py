#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
train_suite_copy.py
Q-learning training suite for pokemon_sim.PokemonBattleEnv (Gen 1–3 via PokéAPI).

- Uses your exact pokemon_sim API (PokemonDatabase, TrainerState, PokemonBattleEnv).
- Forces specific move-sets per scenario (via DB helper).
- Early stopping on rolling win-rate.
- Clear, human-readable replay logs (info['log']).
"""

import argparse
import json
import math
import os
import random
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import pokemon_sim as ps  # your module

# ------------------------- DB loader with caching ----------------------------

PKL_PATH = "pokemon_db_g1g3.pkl"

def load_db() -> ps.PokemonDatabase:
    if os.path.exists(PKL_PATH):
        print(f"[DB] Loading database cache from {PKL_PATH}...")
        try:
            return ps.PokemonDatabase.load_pkl(PKL_PATH)
        except Exception as e:
            print(f"[DB] Cache load failed ({e}); rebuilding from PokéAPI...")
    db = ps.PokemonDatabase(generations=(1, 2, 3))
    db.load_all()
    try:
        db.save_pkl(PKL_PATH)
        print(f"[DB] Saved database cache to {PKL_PATH}")
    except Exception as e:
        print(f"[DB] Warning: failed to save cache ({e})")
    return db

# ------------------------------ Helpers -------------------------------------

def make_mon_with_moves(db: ps.PokemonDatabase, species: str, moves: List[str], level: int = 50) -> ps.PokemonInstance:
    """Create a PokemonInstance with an explicit move list (by names)."""
    species = species.lower()
    if species not in db.species_map:
        raise ValueError(f"Unknown species '{species}' in DB. Did DB.load_all() run?")
    move_objs: List[ps.Move] = []
    for m in moves:
        m = m.lower()
        if m not in db.move_map:
            mv = db._load_move(m)  # use DB's loader
            if mv is None:
                raise ValueError(f"Move '{m}' could not be loaded (maybe SHADOW or unavailable).")
            db.move_map[m] = mv
        move_objs.append(db.move_map[m])
    return ps.PokemonInstance(species=db.species_map[species], level=level, moves=move_objs)

def build_trainer(db: ps.PokemonDatabase, roster: List[Tuple[str, List[str]]], level: int = 50) -> ps.TrainerState:
    team = [make_mon_with_moves(db, name, moves, level) for (name, moves) in roster]
    return ps.TrainerState(team=team, active_index=0)

def state_key(obs: Dict) -> str:
    try:
        return json.dumps(obs, sort_keys=True, default=str)
    except Exception:
        return str(obs)

def binom_ci_95(p: float, n: int) -> Tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    z = 1.96
    denom = 1 + z*z/n
    center = (p + z*z/(2*n)) / denom
    margin = z * math.sqrt((p*(1-p) + z*z/(4*n)) / n) / denom
    return (max(0.0, center - margin), min(1.0, center + margin))

# ---------------------------- Q-learning core --------------------------------

class QTable:
    def __init__(self):
        self.q: Dict[Tuple[str, int], float] = {}

    def get(self, s: str, a: int, default=0.0) -> float:
        return self.q.get((s, a), default)

    def set(self, s: str, a: int, v: float):
        self.q[(s, a)] = v

    def max_a(self, s: str, legal: List[int]) -> Tuple[int, float]:
        if not legal:
            return 0, 0.0
        best_a = legal[0]
        best_v = self.get(s, best_a, 0.0)
        for a in legal[1:]:
            v = self.get(s, a, 0.0)
            if v > best_v:
                best_a, best_v = a, v
        return best_a, best_v

@dataclass
class Scenario:
    key: str
    title: str
    mode: str  # "1v1" or "2v2" (for logging only)
    allow_switch_q: bool
    allow_switch_r: bool
    team1: List[Tuple[str, List[str]]]
    team2: List[Tuple[str, List[str]]]
    episodes: int
    early_stop_window: int = 500
    early_stop_wr: float = 0.70
    epsilon_start: float = 0.50
    epsilon_min: float = 0.05
    epsilon_decay_frac: float = 0.60
    alpha: float = 0.40
    gamma: float = 0.97
    max_turns: int = 50
    level: int = 50

class EpsGreedy:
    def __init__(self, eps_start: float, eps_min: float, decay_frac: float, total_eps: int):
        self.start = eps_start
        self.min = eps_min
        self.decay_until = max(1, int(total_eps * decay_frac))

    def epsilon(self, ep: int) -> float:
        if ep >= self.decay_until:
            return self.min
        span = max(1, self.decay_until - 1)
        return self.min + (self.start - self.min) * (1 - ep / span)

def legal_actions_for_side(env: ps.PokemonBattleEnv, side: int, allow_switch: bool) -> List[int]:
    """Returns action indices that are valid for the given side."""
    trainer = env.trainer1 if side == 1 else env.trainer2
    # Moves (0..3) but clamp to actual # moves
    num_moves = min(4, len(trainer.active_pokemon().moves))
    acts = list(range(num_moves))
    if allow_switch:
        sw = trainer.available_switch_indices()
        # Encode as 4 + idx into "available_switches"
        acts.extend([4 + i for i in range(len(sw))])
    return acts

def random_opponent_action(env: ps.PokemonBattleEnv, allow_switch: bool, switch_prob: float = 0.10) -> int:
    """Random opponent policy similar to pokemon_sim.RandomPolicy."""
    trainer = env.trainer2
    moves_n = min(4, len(trainer.active_pokemon().moves))
    switches = trainer.available_switch_indices() if allow_switch else []
    if switches and random.random() < switch_prob:
        return 4 + random.randint(0, len(switches) - 1)
    else:
        return random.randint(0, max(0, moves_n - 1))

def print_team_once(trainer: ps.TrainerState, tag: str):
    print(f"[TEAM] {tag}:")
    for p in trainer.team:
        print(f"  - {p.species.name}: moves = {[m.name for m in p.moves]}")

def format_replay_turn(turn_no: int, log_lines: List[str]) -> str:
    if not log_lines:
        return f"  t{turn_no:02d}: (no events)"
    out = [f"  t{turn_no:02d}:"]
    for line in log_lines:
        out.append("    " + line)
    return "\n".join(out)

def train_and_eval(db: ps.PokemonDatabase, scn: Scenario, save_prefix: str = "q_"):
    # Build env per episode by cloning initial trainers (the env does this on reset)
    t1_base = build_trainer(db, scn.team1, level=scn.level)
    t2_base = build_trainer(db, scn.team2, level=scn.level)
    env = ps.PokemonBattleEnv(db=db, trainer1=t1_base, trainer2=t2_base)

    print_team_once(t1_base, "Player 1 team")
    print_team_once(t2_base, "Player 2 team")

    qtab = QTable()
    sched = EpsGreedy(scn.epsilon_start, scn.epsilon_min, scn.epsilon_decay_frac, scn.episodes)

    wins_total = 0
    wins_window: List[int] = []
    steps_total = 0
    window = scn.early_stop_window

    replays: List[List[List[str]]] = []  # list of episodes -> per-turn logs
    max_replays = 3

    t0 = time.time()
    for ep in range(1, scn.episodes + 1):
        eps = sched.epsilon(ep)
        obs = env.reset()
        s = state_key(obs)
        done = False
        steps = 0

        capture = ep <= max_replays
        ep_logs: List[List[str]] = []

        while not done and steps < scn.max_turns:
            steps += 1
            # Q-agent action
            legal_q = legal_actions_for_side(env, side=1, allow_switch=scn.allow_switch_q)
            if not legal_q or random.random() < eps:
                a_q = random.choice(legal_q) if legal_q else 0
            else:
                a_q = qtab.max_a(s, legal_q)[0]

            # Opponent action
            a_r = random_opponent_action(env, allow_switch=scn.allow_switch_r)

            obs2, reward, done, info = env.step(a_q, a_r)
            s2 = state_key(obs2)

            # Q update
            next_legal = legal_actions_for_side(env, side=1, allow_switch=scn.allow_switch_q)
            _, maxq_next = qtab.max_a(s2, next_legal) if next_legal else (0, 0.0)
            old = qtab.get(s, a_q, 0.0)
            new = old + scn.alpha * (reward + scn.gamma * maxq_next - old)
            qtab.set(s, a_q, new)
            s = s2

            if capture:
                ep_logs.append(info.get("log", []))

        steps_total += steps
        win = 1 if reward > 0 else 0
        wins_total += win
        wins_window.append(win)
        if len(wins_window) > window:
            wins_window.pop(0)

        if ep == 1 or ep % 200 == 0:
            wr_last = 100.0 * (sum(wins_window) / len(wins_window))
            print(f"[TRAIN] ep {ep:5d}/{scn.episodes:<5d} | ε={eps:.3f} | win(last {len(wins_window):3d})={wr_last:5.1f}% | elapsed={time.time()-t0:.1f}s")

        if capture:
            replays.append(ep_logs)

        if len(wins_window) == window:
            wr_w = sum(wins_window) / window
            if wr_w >= scn.early_stop_wr:
                print(f"[ES] wr={wr_w*100:.1f}% over last {window}; stopping early at ep {ep}.")
                break

    # Save Q-table
    q_path = f"{save_prefix}{scn.key}.json"
    with open(q_path, "w") as f:
        json.dump({f"{k[0]}::{k[1]}": v for k, v in qtab.q.items()}, f)
    # Greedy eval
    eval_eps = 1000
    wins_eval = 0
    steps_eval = 0
    for _ in range(eval_eps):
        obs = env.reset()
        s = state_key(obs)
        done = False
        steps = 0
        while not done and steps < scn.max_turns:
            steps += 1
            legal_q = legal_actions_for_side(env, side=1, allow_switch=scn.allow_switch_q)
            a_q = qtab.max_a(s, legal_q)[0] if legal_q else 0
            a_r = random_opponent_action(env, allow_switch=scn.allow_switch_r)
            obs, reward, done, _ = env.step(a_q, a_r)
            s = state_key(obs)
        steps_eval += steps
        wins_eval += 1 if reward > 0 else 0

    wr = wins_eval / eval_eps
    lo, hi = binom_ci_95(wr, eval_eps)
    print(f"\n[HEADLINE] Win-rate: {wr*100:.1f}%  (95% CI {lo*100:.1f}–{hi*100:.1f}%) | Avg turns: {steps_eval/eval_eps:.2f}")

    print("\n== Replay (samples) ==")
    for i, ep_logs in enumerate(replays, 1):
        print(f"Episode {i} | Turns: {len(ep_logs)}")
        for t, logs in enumerate(ep_logs, 1):
            print(format_replay_turn(t, logs))

# ------------------------------- Scenarios -----------------------------------

# Notes:
# - S0 Mirror 1v1 sanity: same mon + same moves both sides.
# - S1 ensures the opponent can actually hit Zangoose (avoid degenerate immunity).
# - S2/S3 are 2v2 with/without switching (Q can switch in S2, not in S3).
# - Other scenarios mirror the earlier list you wanted.

def S0() -> Scenario:
    return Scenario(
        key="S0",
        title="S0 Mirror 1v1 (sanity): Zangoose (Q) vs Zangoose (Random)",
        mode="1v1",
        allow_switch_q=False,
        allow_switch_r=False,
        team1=[("zangoose", ["shadow-ball", "double-edge", "scratch", "tackle"])],
        team2=[("zangoose", ["shadow-ball", "double-edge", "scratch", "tackle"])],
        episodes=2000
    )

def S1() -> Scenario:
    # Dusclops given non-Ghost coverage so it can hit Normal-type Zangoose
    return Scenario(
        key="S1",
        title="S1 1v1 type-advantage: Zangoose (Q) vs Dusclops (Random)",
        mode="1v1",
        allow_switch_q=False,
        allow_switch_r=False,
        team1=[("zangoose", ["shadow-ball", "double-edge", "scratch", "tackle"])],
        team2=[("dusclops", ["earthquake", "ice-beam", "shadow-ball", "rock-slide"])],
        episodes=2500
    )

def S2() -> Scenario:
    return Scenario(
        key="S2",
        title="S2 2v2 (SWITCH): Gyarados+Dugtrio (Q) vs same (Random)",
        mode="2v2",
        allow_switch_q=True,
        allow_switch_r=True,
        team1=[
            ("gyarados", ["hydro-pump", "waterfall", "tackle", "whirlpool"]),
            ("dugtrio",  ["earthquake", "rock-slide", "mud-slap", "scratch"]),
        ],
        team2=[
            ("gyarados", ["hydro-pump", "waterfall", "tackle", "whirlpool"]),
            ("dugtrio",  ["earthquake", "rock-slide", "mud-slap", "scratch"]),
        ],
        episodes=8000
    )

def S3() -> Scenario:
    return Scenario(
        key="S3",
        title="S3 2v2 (NO-SWITCH): Gyarados+Dugtrio (Q) vs same (Random)",
        mode="2v2",
        allow_switch_q=False,
        allow_switch_r=False,
        team1=[
            ("gyarados", ["hydro-pump", "waterfall", "tackle", "whirlpool"]),
            ("dugtrio",  ["earthquake", "rock-slide", "mud-slap", "scratch"]),
        ],
        team2=[
            ("gyarados", ["hydro-pump", "waterfall", "tackle", "whirlpool"]),
            ("dugtrio",  ["earthquake", "rock-slide", "mud-slap", "scratch"]),
        ],
        episodes=5000
    )

def S4() -> Scenario:
    # Accuracy vs Power: Magneton (Thunder vs Zap Cannon) vs Gyarados
    return Scenario(
        key="S4",
        title="S4 1v1 Accuracy-vs-Power: Magneton (Q) vs Gyarados (Random)",
        mode="1v1",
        allow_switch_q=False,
        allow_switch_r=False,
        team1=[("magneton", ["zap-cannon", "thunder", "tackle", "rage"])],
        team2=[("gyarados",  ["hydro-pump", "waterfall", "tackle", "whirlpool"])],
        episodes=3000
    )

def S5() -> Scenario:
    # Immunity trap: Dugtrio vs Gyarados (EQ is immune; agent should learn Rock Slide/other)
    return Scenario(
        key="S5",
        title="S5 1v1 Immunity Trap: Dugtrio (Q) vs Gyarados (Random)",
        mode="1v1",
        allow_switch_q=False,
        allow_switch_r=False,
        team1=[("dugtrio",  ["earthquake", "rock-slide", "mud-slap", "scratch"])],
        team2=[("gyarados", ["hydro-pump", "waterfall", "tackle", "whirlpool"])],
        episodes=3000
    )

def S6() -> Scenario:
    # Cross-coverage with switching
    return Scenario(
        key="S6",
        title="S6 2v2 Cross-Coverage: Magneton+Dugtrio (Q) vs Gyarados+Magneton (Random)",
        mode="2v2",
        allow_switch_q=True,
        allow_switch_r=True,
        team1=[
            ("magneton", ["zap-cannon", "thunder", "tackle", "rage"]),
            ("dugtrio",  ["earthquake", "rock-slide", "mud-slap", "scratch"]),
        ],
        team2=[
            ("gyarados", ["hydro-pump", "waterfall", "tackle", "whirlpool"]),
            ("magneton", ["zap-cannon", "thunder", "tackle", "rage"]),
        ],
        episodes=7000
    )

def S7() -> Scenario:
    return Scenario(
        key="S7",
        title="S7 2v2 Cross-Coverage (NO-SWITCH)",
        mode="2v2",
        allow_switch_q=False,
        allow_switch_r=False,
        team1=[
            ("magneton", ["zap-cannon", "thunder", "tackle", "rage"]),
            ("dugtrio",  ["earthquake", "rock-slide", "mud-slap", "scratch"]),
        ],
        team2=[
            ("gyarados", ["hydro-pump", "waterfall", "tackle", "whirlpool"]),
            ("magneton", ["zap-cannon", "thunder", "tackle", "rage"]),
        ],
        episodes=6000
    )

def S8() -> Scenario:
    return Scenario(
        key="S8",
        title="S8 1v1 STAB vs Coverage: Zangoose (Q) vs Magneton (Random)",
        mode="1v1",
        allow_switch_q=False,
        allow_switch_r=False,
        team1=[("zangoose",  ["shadow-ball", "double-edge", "scratch", "tackle"])],
        team2=[("magneton", ["zap-cannon", "thunder", "tackle", "rage"])],
        episodes=3000
    )

def S9() -> Scenario:
    return Scenario(
        key="S9",
        title="S9 2v2 Water mirror (chip vs DPS): Gyarados+Gyarados",
        mode="2v2",
        allow_switch_q=True,
        allow_switch_r=True,
        team1=[
            ("gyarados", ["whirlpool", "waterfall", "tackle", "hydro-pump"]),
            ("gyarados", ["whirlpool", "waterfall", "tackle", "hydro-pump"]),
        ],
        team2=[
            ("gyarados", ["whirlpool", "waterfall", "tackle", "hydro-pump"]),
            ("gyarados", ["whirlpool", "waterfall", "tackle", "hydro-pump"]),
        ],
        episodes=6000
    )

_POOL = [
    ("zangoose", ["shadow-ball", "double-edge", "scratch", "tackle"]),
    ("dusclops", ["earthquake", "ice-beam", "shadow-ball", "rock-slide"]),
    ("gyarados", ["hydro-pump", "waterfall", "tackle", "whirlpool"]),
    ("dugtrio",  ["earthquake", "rock-slide", "mud-slap", "scratch"]),
    ("magneton", ["zap-cannon", "thunder", "tackle", "rage"]),
]
def S10(seed: int = 7) -> Scenario:
    rng = random.Random(seed)
    def pick_two():
        a = rng.choice(_POOL)
        b = rng.choice(_POOL)
        while b[0] == a[0]:
            b = rng.choice(_POOL)
        return [a, b]
    return Scenario(
        key="S10",
        title="S10 2v2 Randomized pairs (robustness)",
        mode="2v2",
        allow_switch_q=True,
        allow_switch_r=True,
        team1=pick_two(),
        team2=pick_two(),
        episodes=6000
    )

SCENARIOS = {
    "S0": S0, "S1": S1, "S2": S2, "S3": S3, "S4": S4,
    "S5": S5, "S6": S6, "S7": S7, "S8": S8, "S9": S9, "S10": S10
}

# ----------------------------------- CLI -------------------------------------

def run_one(db: ps.PokemonDatabase, key: str) -> int:
    ctor = SCENARIOS.get(key)
    if ctor is None:
        print(f"Unknown scenario key: {key}")
        return 1
    scn = ctor() if key != "S10" else ctor(7)
    print(f"=== {scn.title} ===")
    train_and_eval(db, scn, save_prefix="q_")
    return 0

def run_all(db: ps.PokemonDatabase) -> int:
    for key in [f"S{i}" for i in range(0, 11)]:
        print("=" * 88)
        rc = run_one(db, key)
        if rc != 0:
            return rc
    return 0

def main():
    ap = argparse.ArgumentParser(description="Q-learning suite for pokemon_sim (S0–S10).")
    ap.add_argument("--only", type=str, default=None, help="Run one scenario key (e.g., S3).")
    ap.add_argument("--list", action="store_true", help="List scenarios and exit.")
    args = ap.parse_args()

    if args.list:
        for k in sorted(SCENARIOS.keys(), key=lambda x: (len(x), x)):
            title = SCENARIOS.title
            print(f"{k}: {title}")
        return

    db = load_db()  # will hit PokéAPI on first run, then cache

    if args.only:
        exit(run_one(db, args.only))
    else:
        exit(run_all(db))

if __name__ == "__main__":
    main()

