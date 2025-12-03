# TRAIN_RL: details

`train_rl.py` trains a **tabular Q-learning** agent for a simple **Gen-3 Pokémon 1v1** battle using the provided simulator `pokemon_sim.py`. The default matchup is:

- **Player 1 (our agent):** Torchic  
- **Player 2 (opponent):** Treecko  
- **Level:** 10  
- **Opponent policy:** **Uniform random** over its legal moves each turn

After training, the script evaluates the **greedy** (exploitation-only) version of the learned policy against the same random opponent, prints headline metrics with confidence intervals, and saves detailed battle logs.

---

## 1) Scope & Current Setup

- **Agent:** Tabular **Q-learning** (model-free).
- **Matchup:** **Torchic (P1)** vs **Treecko (P2)** at **Level 10**.
- **Opponent:** **Uniform-random** policy over its legal moves each turn.
- **Environment:** 1v1 battles driven by `pokemon_sim.py` (Gen-3 data via PokéAPI, simplified damage model).
- **Goal:** Learn a move-selection policy for the P1 species that maximizes win rate against the random opponent.

---

## 2) State, Action, Reward (SAR)

### State (tabular key)
A compact 10×10 grid built from **HP buckets**:
- `s = (p1_hp_bucket, p2_hp_bucket)`
- Bucket rule:
  - `bucket = floor(10 * current_hp / max_hp)` → integer in **{0,…,9}**  
  - 0 ≈ near-faint, 9 ≈ full/near-full
- Species, types, move identities, speed order, stat boosts, etc. are **implicit** (fixed by matchup) and **not encoded** to keep the table small and learning fast.

### Actions
- **P1 legal actions** are indices of its **available moves** (up to 4).  
- We do **not** include switching or items in this starter setup.

### Reward shaping (from `pokemon_sim.py`)
- **Intermediate reward:** proportional to **HP difference** each step (simulator’s shaping).
- **Terminal reward:** +1 (win), −1 (loss), 0 (draw/timeout).  
- **Win/loss** for metrics is taken from the **sign of the final reward**.

---

## 3) Q-Learning Algorithm

### Behavior policy (training)
- **ε-greedy** over legal actions:
  - With probability **ε**: pick a random legal move.
  - Else: pick the **argmax-Q** legal move.

### Update (tabular)
For transition `(s, a, r, s')`:
\[
Q(s,a)\leftarrow Q(s,a) + \alpha\Big(r + \gamma \max_{a'\in \mathcal{A}(s')} Q(s',a') - Q(s,a)\Big)
\]
- `α` (ALPHA): learning rate
- `γ` (GAMMA): discount
- `ε`: decays **linearly** from `EPS_START` to `EPS_END` over `EPS_DECAY_EPISODES`.

### Policy (evaluation)
- **Greedy** (no exploration).

---

## 4) Training Loop — Episode Anatomy

Each episode:
1. Create fresh environment `make_starter_battle_env(db, level=10)` and `reset()`.
2. Compute `s = (bucketize(p1.hp), bucketize(p2.hp))`.
3. Choose `a ~ ε-greedy` over the **current** legal moves for P1.
4. Opponent chooses `a₂ ~ UniformRandom` over P2 legal moves.
5. Step environment → `(obs, r, done, info)` and build `s'` via bucketing.
6. **Q-update** with the reward and `max_a' Q(s', a')` (legal in `s'`).
7. Terminate if `done` or `MAX_TURNS`.
8. Record the episode outcome by the sign of the terminal reward.

**Progress logs** (every `LOG_EVERY` episodes): episode idx, ε, smoothed training win rate over `SMOOTH_W`, elapsed, ETA.

**Optional early stopping** (if enabled in your code): periodic short eval; stop if win rate reaches a target for N consecutive checks.

---

## 5) Evaluation & Outputs

After training, the script runs a **greedy** policy vs random for `EVAL_EPISODES` and produces:

- **Console headline** with **Wilson 95% CI**:
  - Win‐rate, avg turns, avg KO diff, elapsed time.
- **Q-table JSON** (if `SAVE_Q_PATH` is set):  
  - Keys look like `"[p1_bucket,p2_bucket]|action_index"` → Q-value (float).
- **Episode-level CSV** (if implemented in your local copy):  
  - `eval_summary.csv`:
    ```
    episode,p1,p2,result,turns,terminal_r
    1,torchic,treecko,win,6,1.0
    ...
    ```
- **Replay logs** subset (if implemented):  
  - `eval_battles_sample.jsonl` with detailed turn logs for the first `EVAL_LOG_EPISODES`.

### Wilson 95% CI (used for headline)
For `w` wins out of `n`:
\[
\hat{p} = w/n,\quad
\text{CI} = \frac{\hat{p} + \frac{z^2}{2n} \pm z \sqrt{\frac{\hat{p}(1-\hat{p})}{n} + \frac{z^2}{4n^2}}}{1 + \frac{z^2}{n}},\; z=1.96
\]

---

## 6) Configuration Knobs (edit in `train_rl.py`)

- **Random seed:** `SEED`
- **Level:** `LEVEL`
- **Episodes:** `EPISODES` (e.g., **800** default in faster configs)
- **Max turns per episode:** `MAX_TURNS` (e.g., 30)
- **Learning rate / discount:** `ALPHA`, `GAMMA`
- **Exploration:** `EPS_START`, `EPS_END`, `EPS_DECAY_EPISODES`
- **Logging:** `LOG_EVERY`, `SMOOTH_W`
- **Early stopping (if present in your copy):** target win rate, check cadence, patience
- **Evaluation:** `EVAL_EPISODES`, optional `EVAL_LOG_EPISODES`
- **Artifacts:** `SAVE_Q_PATH` (set to `None` to disable saving Q)

> **Tip:** For quick iteration, start with `EPISODES ≈ 400–800`, `MAX_TURNS ≈ 25–30`, and **enable early stopping** if your branch supports it.

---

## 7) How to Change the Matchup

- Replace `make_starter_battle_env(db, level=LEVEL)` with a helper that builds **any** 1v1:
  ```python
  torchic = db.create_pokemon_instance("torchic", level=LEVEL)
  treecko = db.create_pokemon_instance("treecko", level=LEVEL)
  env = PokemonBattleEnv(db, TrainerState([torchic], 0), TrainerState([treecko], 0))

# Pseudocode
Initialize Q[(hp_b1, hp_b2), a] = 0
for episode in 1..EPISODES:
    env ← Torchic vs Treecko (level 10), reset()
    s ← (bucket(p1.hp), bucket(p2.hp))
    for t in 1..MAX_TURNS:
        a  ← ε-greedy over legal(s); a2 ← random legal
        _, r, done, _ ← env.step(a, a2)
        s' ← (bucket(p1.hp), bucket(p2.hp))
        Q[s,a] ← Q[s,a] + α ( r + γ max_{a'∈legal(s')} Q(s',a') − Q[s,a] )
        s ← s'
        if done: break
Evaluate greedy(Q) vs random for EVAL_EPISODES; report win-rate + Wilson 95% CI,
and save per-episode outcomes + sample replay logs (if enabled).

