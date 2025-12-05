# Pokémon Gen-3 RL Training Suite

A compact research suite for learning move selection and basic switching in simplified Gen-3 style Pokémon battles. It uses:

* `pokemon_sim.py` — a 1v1-per-side battle environment backed by **PokéAPI** (Gen 1–3) with a consistent damage model.
* `train_suite_full.py` — a plug-and-play **Q-learning** trainer/evaluator with a curated set of scenarios (S0–S10) that probe type matchups, immunities, accuracy vs. power, and the value of switching.

The suite trains a Q-agent vs. a random baseline and reports win rates with 95% CIs and short, readable replays.

---

## Quick Start

```bash
# (Optional) create and activate a venv
python3 -m venv .venv
source .venv/bin/activate

# Dependencies for PokéAPI access
pip install requests

# First run fetches Gen 1–3 data from PokéAPI and caches it
python3 train_suite_full.py --list          # list scenarios
python3 train_suite_full.py --only S0       # run one scenario
python3 train_suite_full.py                 # run S0–S10
```

**Outputs**

* Trained tables: `q_Si.json` per scenario.
* Console: training progress, headline metrics, human-readable replays.
* First run creates `pokemon_db_g1g3.pkl` (local cache of species/moves/type chart).

---

## Files

* **`pokemon_sim.py`**
  Loads Gen 1–3 Pokémon and moves (PokéAPI), builds the type chart, and exposes:

  * `PokemonDatabase` (species/moves, type multipliers, team/move creation)
  * `PokemonBattleEnv` (1v1 battle loop, accuracy, priority, speed order, crit, STAB, typing)
  * Simple policies (random, greedy power, greedy type) for manual tests

* **`train_suite_full.py`**
  Q-learning trainer with scenario definitions (S0–S10), early stopping, evaluation, and readable replay printing.

---

## Action Space

In `PokemonBattleEnv`:

* `0..3` → use move at that index (clamped to the Pokémon’s move count)
* `4 + i` → switch to the *i-th available* bench slot (if switching is allowed and target is not fainted)

---

## Rewards and Termination

* While both sides have available Pokémon:
  `reward = (sum(P1_HP) − sum(P2_HP)) / 100.0`  (dense shaping)
* Terminal rewards: `+1` win, `−1` loss, `0` double-faint
* Turn cap: `max_turns = 50` to avoid loops
* Faints auto-trigger a forced switch if teammates remain

---

## Q-Learning

Update rule:

```
Q(s,a) ← Q(s,a) + α [ r + γ max_{a’} Q(s’, a’) − Q(s,a) ]
```

* **Policy**: ε-greedy over the Q-table (agent) vs. **random** opponent (switches occasionally if allowed)
* **Exploration**: linear decay from `epsilon_start` → `epsilon_min` over first `epsilon_decay_frac` of episodes
* **Early stopping**: rolling win-rate over last `early_stop_window` episodes; stop when ≥ `early_stop_wr`

**Default hyperparameters** (scenario can override):

* `episodes` per scenario: see table below
* `epsilon_start = 0.50`
* `epsilon_min   = 0.05`
* `epsilon_decay_frac = 0.60`
* `alpha (step size) = 0.40`
* `gamma (discount)  = 0.97`
* `max_turns = 50`
* Early stop: `early_stop_window = 500`, `early_stop_wr = 0.70`
* Level: 50 for all Pokémon
* Opponent random switch probability: `0.10` (only in switch-enabled scenarios)

---

## State Representation (“Buckets”)

The current implementation uses the full observation dict serialized to a JSON key:

* Active P1/P2: name, HP/max HP, types, base stats (atk/def/spa/spd/spe)
* Team status arrays for both sides (name, HP/max HP), enabling learned switching

**No explicit discretization** is applied. For generalization across similar states, you can optionally add:

* HP buckets (e.g., 0; (0,25]; (25,50]; (50,75]; (75,100])
* Best-move type effectiveness class (0, 0.5, 1, 2)
* Binary flags for “my best move is immune” or “opponent super-effective threat”
* Last action / last damage bucket

---

## Scenario Catalog (S0–S10)

*All Pokémon are **level 50**. Unless noted, each side has one active Pokémon per team slot. “Switching: On” means voluntary switching is allowed; fainting always allows a forced switch.*

### S0 — Mirror 1v1 (Sanity)

* **Switching:** Off
* **Episodes:** 2000
* **Teams:**

  * **Player 1 (Q):** Zangoose — moves: `shadow-ball, double-edge, scratch, tackle`
  * **Player 2 (Random):** Zangoose — same moves
* **Purpose:** Symmetric baseline. Agent should quickly prefer higher-value attacks and stabilize at a high win rate.

---

### S1 — Type Advantage + Immunity Pitfalls

* **Switching:** Off
* **Episodes:** 2500
* **Teams:**

  * **Player 1 (Q):** Zangoose — moves: `shadow-ball, double-edge, scratch, tackle`
  * **Player 2 (Random):** Dusclops — moves: `earthquake, ice-beam, shadow-ball, rock-slide`
* **Purpose:** Q should learn **Ghost hits Ghost** and **Normal can whiff vs Ghost**; coverage on Dusclops prevents degenerate “no effect” loops.

---

### S2 — 2v2 With Switching (Coverage Alignment)

* **Switching:** On
* **Episodes:** 8000
* **Teams:**

  * **Player 1 (Q):**

    * Gyarados — `hydro-pump, waterfall, tackle, whirlpool`
    * Dugtrio — `earthquake, rock-slide, mud-slap, scratch`
  * **Player 2 (Random):** Same pair and moves
* **Purpose:** Learn to **keep favorable matchups** via switching (Water vs Ground) and close out efficiently.

---

### S3 — 2v2 Without Switching (Stuck Leads)

* **Switching:** Off
* **Episodes:** 5000
* **Teams:** Same species/moves as S2 for both sides
* **Purpose:** Baseline against S2 to show the value of switching. Expect **lower** win rate vs S2.

---

### S4 — Accuracy vs. Power (Decision Under Uncertainty)

* **Switching:** Off
* **Episodes:** 3000
* **Teams:**

  * **Player 1 (Q):** Magneton — `zap-cannon, thunder, tackle, rage`
  * **Player 2 (Random):** Gyarados — `hydro-pump, waterfall, tackle, whirlpool`
* **Purpose:** Tradeoff between **huge power / low accuracy** (Zap Cannon) and **high power / better accuracy** (Thunder). Q should track expected damage, not just base power.

---

### S5 — Immunity Trap (Ground vs Flying)

* **Switching:** Off
* **Episodes:** 3000
* **Teams:**

  * **Player 1 (Q):** Dugtrio — `earthquake, rock-slide, mud-slap, scratch`
  * **Player 2 (Random):** Gyarados — `hydro-pump, waterfall, tackle, whirlpool`
* **Purpose:** Learn to **avoid Earthquake** into a Flying target; prefer **Rock Slide** for coverage.

---

### S6 — 2v2 Cross-Coverage With Switching

* **Switching:** On
* **Episodes:** 7000
* **Teams:**

  * **Player 1 (Q):**

    * Magneton — `zap-cannon, thunder, tackle, rage`
    * Dugtrio — `earthquake, rock-slide, mud-slap, scratch`
  * **Player 2 (Random):**

    * Gyarados — `hydro-pump, waterfall, tackle, whirlpool`
    * Magneton — `zap-cannon, thunder, tackle, rage`
* **Purpose:** **Pivot** to maintain favorable pairings (e.g., Dugtrio ↔ Magneton, Magneton ↔ Gyarados). Tests matchup cycling.

---

### S7 — 2v2 Cross-Coverage Without Switching

* **Switching:** Off
* **Episodes:** 6000
* **Teams:** Same species/moves as S6 for both sides
* **Purpose:** Baseline vs S6 to isolate switching’s impact on WR and average turns.

---

### S8 — STAB vs. Coverage (Tough Matchup)

* **Switching:** Off
* **Episodes:** 3000
* **Teams:**

  * **Player 1 (Q):** Zangoose — `shadow-ball, double-edge, scratch, tackle`
  * **Player 2 (Random):** Magneton — `zap-cannon, thunder, tackle, rage`
* **Purpose:** Steel resists Normal and Ghost in Gen-3. Q must pick the **least bad** option; tests convergence in an unfavorable matchup.

---

### S9 — 2v2 Water Mirror (Chip vs. DPS)

* **Switching:** On
* **Episodes:** 6000
* **Teams:**

  * **Player 1 (Q):**

    * Gyarados — `whirlpool, waterfall, tackle, hydro-pump`
    * Gyarados — `whirlpool, waterfall, tackle, hydro-pump`
  * **Player 2 (Random):** Same duo
* **Purpose:** Explore whether Q values **chip** (`whirlpool`) vs **burst/DPS** (`waterfall` / `hydro-pump`) and sequences sensibly.

---

### S10 — 2v2 Randomized Pairs (Robustness)

* **Switching:** On
* **Episodes:** 6000
* **Teams:** Each side randomly draws two distinct Pokémon from
  `{Zangoose, Dusclops, Gyarados, Dugtrio, Magneton}` with the move sets defined above.
* **Purpose:** Light robustness: generalize tabular Q to **unseen pairings** (no function approximator).

---                               |

> **Note:** All Pokémon are level 50. Switching “On” means both sides can voluntarily switch; fainting always allows a forced switch if bench exists.

---

## Running Specific Scenarios

```bash
# List all scenarios
python3 train_suite_full.py --list

# Run a single scenario
python3 train_suite_full.py --only S4

# Run the full battery S0–S10
python3 train_suite_full.py
```

The script prints:

* Per-scenario team summaries (species and moves)
* Training progress with ε, rolling win-rate, early-stop trigger
* Final headline: win-rate with 95% CI, average turns
* **Readable replays** for the first few episodes, e.g.:

```
== Replay (sample) ==
Episode 1 | Result: win | Turns: 7
Player 1's gyarados used waterfall and dealt 40 damage to gyarados (HP: 105/145). It's not very effective...
Player 2's gyarados used hydro-pump and dealt 21 damage to gyarados (HP: 98/145). It's not very effective...
...
Player 1's gyarados used waterfall and dealt 520 damage to dugtrio (HP: 0/85). A critical hit! It's super effective!
Player 2's dugtrio fainted!
```

---

## Extending the Suite

* Add a new `Scenario` (teams, switch flag, episodes) and append it to `SCENARIOS`.
* To improve generalization, introduce **bucketing** in `state_key` (HP bins, matchup flags, last action).
* Swap the random opponent for a stronger baseline (e.g., greedy type/power) by modifying the opponent policy in the training loop.

---

## Notes and Constraints

* Mechanics are simplified but consistent: no abilities, items, weather, stat stages, status, or PP.
* Moves are explicitly set for each scenario to surface the intended lesson and avoid no-effect dead ends.
* First run fetches data from PokéAPI; subsequent runs hit the local cache.

---

## License and Attribution

* Pokémon names and move data: **PokéAPI** — [https://pokeapi.co/](https://pokeapi.co/)
* Educational/research use only.
