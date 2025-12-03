# train_rl.py — Faster Q-learning on the Gen-3 starter battle (Torchic vs Treecko)
# Works with the provided pokemon_sim.py (no changes required)

import csv
import json
import random
import time
from collections import defaultdict, deque
from math import sqrt
from typing import Any, Dict, List, Tuple, Optional

from pokemon_sim import (
    PokemonDatabase,
    PokemonBattleEnv,
    TrainerState,
    make_starter_battle_env,
)

# =========================
# Config
# =========================

SEED = 67
LEVEL = 10

# Q-learning hyperparameters
ALPHA = 0.5
GAMMA = 0.99

# Faster exploration schedule
EPS_START = 0.30
EPS_END   = 0.05
EPS_DECAY_EPISODES = 300   # linear decay horizon for epsilon

# Fewer episodes + short turns
EPISODES = 800
MAX_TURNS = 30

# Progress / logging
LOG_EVERY = 100           # print progress every N episodes
SMOOTH_W = 100            # moving window for training win-rate

# Early stopping
ES_CHECK_EVERY = 100      # evaluate every N episodes
ES_EVAL_EPISODES = 200    # rollouts per eval
ES_TARGET_WR = 0.90       # stop when >= 90% twice in a row
ES_PATIENCE = 2

# Final evaluation
EVAL_EPISODES = 500

# Evaluation outputs
EVAL_SUMMARY_CSV = "eval_summary.csv"            # per-episode outcomes
EVAL_LOGS_JSONL  = "eval_battles_sample.jsonl"   # full logs for first N episodes
EVAL_LOG_EPISODES = 10                           # how many episodes to save full logs for
EVAL_PRINT_FIRST  = 2                            # how many battles to print to console

# Optional: save learned Q-table to disk
SAVE_Q_PATH: Optional[str] = "starter_q_table.json"


# =========================
# Utilities
# =========================

def set_seed(seed: int) -> None:
    random.seed(seed)

def wilson_ci(successes: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = successes / float(n)
    denom = 1.0 + (z * z) / float(n)
    centre = p + (z * z) / (2.0 * n)
    pm = z * sqrt(p * (1.0 - p) / float(n) + (z * z) / (4.0 * n * n))
    lo = (centre - pm) / denom
    hi = (centre + pm) / denom
    return (max(0.0, lo), min(1.0, hi))

def format_eta(seconds_left: float) -> str:
    if seconds_left < 0:
        seconds_left = 0
    m, s = divmod(int(seconds_left), 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h:d}h {m:02d}m {s:02d}s"
    if m > 0:
        return f"{m:d}m {s:02d}s"
    return f"{s:d}s"


# =========================
# State & action adapters
# =========================

def state_key(env: PokemonBattleEnv) -> Tuple[int, int]:
    """
    10 HP buckets (0..9) for each side, fixed matchup (species/types implicit)
    """
    p1 = env.trainer1.active_pokemon()
    p2 = env.trainer2.active_pokemon()

    def bucket(hp, max_hp, n=10):
        if max_hp <= 0:
            return 0
        b = int(n * (hp / float(max_hp)))
        return max(0, min(n - 1, b))

    b1 = bucket(p1.current_hp, p1.max_hp, n=10)
    b2 = bucket(p2.current_hp, p2.max_hp, n=10)
    return (b1, b2)

def legal_actions(env: PokemonBattleEnv, for_player: int = 1) -> List[int]:
    trainer = env.trainer1 if for_player == 1 else env.trainer2
    n = min(4, len(trainer.active_pokemon().moves))
    return list(range(n)) if n > 0 else [0]

def opponent_random_action(env: PokemonBattleEnv) -> int:
    acts = legal_actions(env, for_player=2)
    return random.choice(acts) if acts else 0


# =========================
# Tabular Q-learning
# =========================

class TabularQ:
    def __init__(
        self,
        alpha: float = 0.5,
        gamma: float = 0.99,
        eps_start: float = 1.0,
        eps_end: float = 0.05,
        eps_decay_episodes: int = 1000,
        init_q: float = 0.0,
    ):
        self.Q: Dict[Tuple[Tuple[Any, ...], int], float] = defaultdict(lambda: init_q)
        self.alpha = alpha
        self.gamma = gamma
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay_episodes = max(1, eps_decay_episodes)
        self.episode = 0

    def epsilon(self) -> float:
        # Linear decay across EPS_DECAY_EPISODES
        frac = min(1.0, self.episode / float(self.eps_decay_episodes))
        return self.eps_start + (self.eps_end - self.eps_start) * frac

    def value(self, s, a) -> float:
        return self.Q[(s, a)]

    def set_value(self, s, a, v) -> None:
        self.Q[(s, a)] = v

    def best_action(self, s, legal: List[int]) -> int:
        if not legal:
            return 0
        return max(legal, key=lambda a: self.value(s, a))

    def act_eps_greedy(self, s, legal: List[int]) -> int:
        if not legal:
            return 0
        if random.random() < self.epsilon():
            return random.choice(legal)
        return self.best_action(s, legal)

    def update(self, s, a, r, sp, legal_sp: List[int]) -> None:
        qsa = self.value(s, a)
        max_next = max((self.value(sp, ap) for ap in legal_sp), default=0.0)
        target = r + self.gamma * max_next
        self.set_value(s, a, qsa + self.alpha * (target - qsa))


# =========================
# Training & Evaluation
# =========================

def train_q_on_starter_battle(
    db: PokemonDatabase,
    episodes: int = EPISODES,
    max_turns: int = MAX_TURNS,
    alpha: float = ALPHA,
    gamma: float = GAMMA,
    eps_start: float = EPS_START,
    eps_end: float = EPS_END,
) -> TabularQ:
    """
    Train Q-learning where Player 1 uses Torchic and Player 2 uses Treecko
    (as produced by make_starter_battle_env). Opponent acts uniformly at random.
    """
    q = TabularQ(alpha=alpha, gamma=gamma, eps_start=eps_start,
                 eps_end=eps_end, eps_decay_episodes=EPS_DECAY_EPISODES,
                 init_q=0.0)

    t0 = time.time()
    recent = deque(maxlen=SMOOTH_W)
    wins_total = 0
    passes = 0  # early-stopping consecutive passes counter

    for ep in range(1, episodes + 1):
        q.episode = ep

        env = make_starter_battle_env(db, level=LEVEL)
        env.reset()

        done = False
        turns = 0
        terminal_r = 0.0

        while not done and turns < max_turns:
            s = state_key(env)
            a = q.act_eps_greedy(s, legal_actions(env, 1))
            a2 = opponent_random_action(env)
            _, r, done, _ = env.step(a, a2)
            sp = state_key(env)
            # Note: we pass legal actions for player 1 again; both sides are 1v1
            q.update(s, a, r, sp, legal_actions(env, 1))
            turns += 1
            terminal_r = r

        won = 1 if terminal_r > 0 else 0
        wins_total += won
        recent.append(won)

        if ep % LOG_EVERY == 0 or ep == 1:
            elapsed = time.time() - t0
            rate = ep / max(1.0, elapsed)  # ep/sec
            remaining = episodes - ep
            eta = format_eta(remaining / max(1e-6, rate))
            win_recent = sum(recent) / max(1, len(recent))
            print(
                f"[TRAIN] ep {ep:5d}/{episodes} | "
                f"ε={q.epsilon():.3f} | "
                f"win(last {len(recent):>3d})={100*win_recent:5.1f}% | "
                f"elapsed={elapsed:6.1f}s | ETA~{eta}"
            )

        # ---- Early stopping ----
        if ep % ES_CHECK_EVERY == 0:
            metrics, _, _ = evaluate_policy_on_starter(
                db, q, episodes=ES_EVAL_EPISODES, max_turns=max_turns,
                save_summary_csv=None, save_logs_jsonl=None,
                log_episodes=0, print_first=0
            )
            wr = metrics["win_rate"]
            lo, hi = metrics["ci95"]
            print(f"[ES] wr={100*wr:4.1f}% (95% CI {100*lo:4.1f}–{100*hi:4.1f}%) after {ep} eps")
            if wr >= ES_TARGET_WR:
                passes += 1
            else:
                passes = 0
            if passes >= ES_PATIENCE:
                print(f"[ES] Early stop at ep {ep} (≥ {int(ES_TARGET_WR*100)}% twice).")
                break

    elapsed = time.time() - t0
    print(f"[TRAIN] Complete: {ep} episodes in {elapsed:.1f}s "
          f"(avg {(elapsed/ep):.3f}s/ep). "
          f"Overall training win-rate {100*(wins_total/max(1, ep)):.1f}%")
    return q


def greedy_action(q: TabularQ, s_key: Tuple[Any, ...], legal: List[int]) -> int:
    if not legal:
        return 0
    best_a = legal[0]
    best_q = float("-inf")
    for a in legal:
        v = q.value(s_key, a)
        if v > best_q:
            best_q, best_a = v, a
    return best_a


def evaluate_policy_on_starter(
    db: PokemonDatabase,
    q: TabularQ,
    episodes: int = 1000,
    max_turns: int = 30,
    save_summary_csv: Optional[str] = EVAL_SUMMARY_CSV,
    save_logs_jsonl: Optional[str] = EVAL_LOGS_JSONL,
    log_episodes: int = EVAL_LOG_EPISODES,
    print_first: int = EVAL_PRINT_FIRST,
):
    """
    Evaluates the learned policy vs a RANDOM opponent.
    Returns (metrics, summary_rows, log_records).
      - metrics: dict with win_rate, CI, mean turns, etc.
      - summary_rows: per-episode summaries (list of dicts)
      - log_records: first `log_episodes` full logs (list of dicts)
    Also writes:
      - CSV summary for all episodes (if save_summary_csv not None)
      - JSONL logs for the first `log_episodes` (if save_logs_jsonl not None)
    """
    t0 = time.time()
    wins = 0
    total_turns = 0
    ko_diff_sum = 0  # proxy: +1 win, -1 loss (1v1)

    summary_rows: List[Dict[str, Any]] = []
    log_records: List[Dict[str, Any]] = []

    for ep in range(1, episodes + 1):
        env = make_starter_battle_env(db, level=LEVEL)
        env.reset()
        done = False
        turns = 0
        terminal_r = 0.0
        episode_log: List[str] = []

        p1_name = env.initial_trainer1.team[0].species.name
        p2_name = env.initial_trainer2.team[0].species.name

        while not done and turns < max_turns:
            s = state_key(env)
            a = greedy_action(q, s, legal_actions(env, 1))
            a2 = opponent_random_action(env)
            _, r, done, info = env.step(a, a2)
            turns += 1
            terminal_r = r

            # collect turn-by-turn messages
            if info and "log" in info:
                episode_log.extend(info["log"])

        total_turns += turns

        # determine result
        if terminal_r > 0:
            wins += 1
            ko_diff_sum += 1
            result = "win"
        elif terminal_r < 0:
            ko_diff_sum -= 1
            result = "loss"
        else:
            result = "draw_or_timeout"

        summary_rows.append({
            "episode": ep,
            "p1": p1_name,
            "p2": p2_name,
            "result": result,
            "turns": turns,
            "terminal_r": terminal_r,
        })

        if len(log_records) < max(0, log_episodes):
            log_records.append({
                "episode": ep,
                "p1": p1_name,
                "p2": p2_name,
                "result": result,
                "turns": turns,
                "terminal_r": terminal_r,
                "log": episode_log,
            })

    wr = wins / float(episodes) if episodes else 0.0
    lo, hi = wilson_ci(wins, episodes)
    elapsed = time.time() - t0

    metrics = {
        "episodes": episodes,
        "win_rate": wr,
        "ci95": (lo, hi),
        "avg_turns": total_turns / max(1, episodes),
        "avg_ko_diff": ko_diff_sum / max(1, episodes),
        "elapsed_s": elapsed,
    }

    # ---- Save files ----
    if save_summary_csv:
        with open(save_summary_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["episode", "p1", "p2", "result", "turns", "terminal_r"])
            writer.writeheader()
            writer.writerows(summary_rows)
        print(f"[EVAL] Wrote per-episode summary to {save_summary_csv} ({len(summary_rows)} rows)")

    if save_logs_jsonl and log_records:
        with open(save_logs_jsonl, "w") as f:
            for rec in log_records:
                f.write(json.dumps(rec) + "\n")
        print(f"[EVAL] Wrote detailed logs for first {len(log_records)} episodes to {save_logs_jsonl}")

    # ---- Console sample ----
    if print_first > 0 and log_records:
        k = min(print_first, len(log_records))
        for i in range(k):
            rec = log_records[i]
            print("\n== Replay (sample) ==")
            print(f"Episode {rec['episode']} | {rec['p1']} (Q) vs {rec['p2']} (Random) | "
                  f"Result: {rec['result']} | Turns: {rec['turns']}")
            for line in rec["log"]:
                print(line)

    return metrics, summary_rows, log_records


# =========================
# Main
# =========================

def main():
    set_seed(SEED)

    print("Loading Gen 3 data from PokéAPI...")
    db = PokemonDatabase()
    db.load_all()

    print("\n=== Train Q-learning on STARTER battle (Torchic vs Treecko) ===")
    q = train_q_on_starter_battle(
        db,
        episodes=EPISODES,
        max_turns=MAX_TURNS,
        alpha=ALPHA,
        gamma=GAMMA,
        eps_start=EPS_START,
        eps_end=EPS_END,
    )

    if SAVE_Q_PATH:
        print(f"Saving Q-table to {SAVE_Q_PATH} ...")
        flat = {json.dumps(s, separators=(',', ':')) + f"|{a}": v for (s, a), v in q.Q.items()}
        with open(SAVE_Q_PATH, "w") as f:
            json.dump(flat, f)
        print(f"Saved {len(flat)} (state,action) entries.")

    print("\n=== Evaluate greedy policy vs RANDOM opponent ===")
    metrics, _, _ = evaluate_policy_on_starter(
        db, q,
        episodes=EVAL_EPISODES,
        max_turns=MAX_TURNS,
        save_summary_csv=EVAL_SUMMARY_CSV,
        save_logs_jsonl=EVAL_LOGS_JSONL,
        log_episodes=EVAL_LOG_EPISODES,
        print_first=EVAL_PRINT_FIRST,
    )

    lo, hi = metrics["ci95"]
    print(
        f"\n[HEADLINE] Win-rate: {100*metrics['win_rate']:.1f}%  "
        f"(95% CI {100*lo:.1f}–{100*hi:.1f}%) | "
        f"Avg turns: {metrics['avg_turns']:.2f} | "
        f"Avg KO diff: {metrics['avg_ko_diff']:+.2f} | "
        f"time: {metrics['elapsed_s']:.1f}s"
    )

if __name__ == "__main__":
    main()

