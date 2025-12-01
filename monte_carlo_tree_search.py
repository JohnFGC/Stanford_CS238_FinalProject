import random
import numpy as np
from collections import defaultdict
from pokemon_sim import PokemonDatabase, PokemonBattleEnv, TrainerState, MoveCategory, PokemonType

# =========================
# Hyperparameters
# =========================
ALPHA = 0.1       # learning rate
GAMMA = 0.99      # discount factor
EPSILON = 0.1     # epsilon-greedy exploration
EPISODES = 500    # episodes per Pokémon

# =========================
# Utilities
# =========================
def flatten_obs(obs):
    """
    Convert obs dict to a simple, hashable state representation for tabular Q-learning.
    We'll use:
        - active Pokémon HP %
        - opponent Pokémon HP %
        - active Pokémon speed
        - opponent Pokémon speed
    """
    p1 = obs["p1_active"]
    p2 = obs["p2_active"]

    p1_hp_pct = int((p1["hp"] / p1["max_hp"]) * 10)
    p2_hp_pct = int((p2["hp"] / p2["max_hp"]) * 10)
    p1_spe = int(p1["spe"] / 10)
    p2_spe = int(p2["spe"] / 10)

    return (p1_hp_pct, p2_hp_pct, p1_spe, p2_spe)

# =========================
# Q-Learning Loop
# =========================
def train_q_learning_for_species(db, species_name):
    print(f"Training Q-Learning policy for {species_name}...")
    Q = defaultdict(lambda: np.zeros(4))  # 4 moves per Pokémon

    for episode in range(EPISODES):
        # Create same Pokémon for both sides
        p1 = db.create_pokemon_instance(species_name, level=50)
        p2 = db.create_pokemon_instance(species_name, level=50)

        t1 = TrainerState(team=[p1])
        t2 = TrainerState(team=[p2])

        env = PokemonBattleEnv(db, t1, t2)
        obs = env.reset()
        done = False

        while not done:
            state = flatten_obs(obs)
            # epsilon-greedy action
            if random.random() < EPSILON:
                action = random.randint(0, len(p1.moves)-1)
            else:
                action = int(np.argmax(Q[state]))

            opp_action = random.randint(0, len(p2.moves)-1)

            next_obs, reward, done, info = env.step(action, opp_action)
            next_state = flatten_obs(next_obs)

            # Q-Learning update
            Q[state][action] = Q[state][action] + ALPHA * (
                reward + GAMMA * np.max(Q[next_state]) - Q[state][action]
            )

            obs = next_obs

        if (episode + 1) % 50 == 0:
            print(f"Episode {episode + 1}/{EPISODES} completed.")

    return Q

# =========================
# Evaluate Learned Policy
# =========================
def evaluate_policy(Q, db, species_name, trials=20):
    wins = 0
    for _ in range(trials):
        p1 = db.create_pokemon_instance(species_name, level=50)
        p2 = db.create_pokemon_instance(species_name, level=50)
        t1 = TrainerState(team=[p1])
        t2 = TrainerState(team=[p2])
        env = PokemonBattleEnv(db, t1, t2)
        obs = env.reset()
        done = False

        while not done:
            state = flatten_obs(obs)
            action = int(np.argmax(Q[state]))
            opp_action = random.randint(0, len(p2.moves)-1)
            obs, reward, done, info = env.step(action, opp_action)

        if reward > 0:
            wins += 1

    win_rate = wins / trials
    print(f"Policy win rate for {species_name}: {win_rate*100:.1f}%")
    return win_rate

# =========================
# Main
# =========================
if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)

    db = PokemonDatabase.load_pkl("gen3_db.pkl")

    # Example: train and evaluate for Torchic
    species_list = ["torchic", "treecko", "mudkip"]  # can loop over all db.species_map.keys()
    for species_name in species_list:
        Q_policy = train_q_learning_for_species(db, species_name)
        evaluate_policy(Q_policy, db, species_name)
