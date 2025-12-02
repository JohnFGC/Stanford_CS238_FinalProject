"""
Pokemon Gen 3 Battle Simulator using PokéAPI

- Downloads Gen 3 Pokémon and moves from https://pokeapi.co/
- Builds a type chart and move database
- Provides a PokemonBattleEnv for RL / planning

Requirements:
    pip install requests
"""

import enum
import math
import random
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple, Any

import requests
import pickle

POKEAPI_BASE = "https://pokeapi.co/api/v2"


# =========================
# Types & Type Effectiveness
# =========================

class PokemonType(enum.Enum):
    NORMAL = "normal"
    FIRE = "fire"
    WATER = "water"
    GRASS = "grass"
    ELECTRIC = "electric"
    ICE = "ice"
    FIGHTING = "fighting"
    POISON = "poison"
    GROUND = "ground"
    FLYING = "flying"
    PSYCHIC = "psychic"
    BUG = "bug"
    ROCK = "rock"
    GHOST = "ghost"
    DRAGON = "dragon"
    DARK = "dark"
    STEEL = "steel"
    FAIRY = "fairy"  # not Gen3 but in API; harmless
    SHADOW = "shadow"


class MoveCategory(enum.Enum):
    PHYSICAL = "physical"
    SPECIAL = "special"
    STATUS = "status"


@dataclass
class Move:
    name: str
    move_type: PokemonType
    power: Optional[int]  # can be None for some moves
    accuracy: Optional[float]  # 0..1 or None
    category: MoveCategory
    priority: int = 0


@dataclass
class PokemonSpecies:
    name: str
    types: List[PokemonType]
    base_hp: int
    base_atk: int
    base_def: int
    base_spa: int
    base_spd: int
    base_spe: int


@dataclass
class PokemonInstance:
    species: PokemonSpecies
    level: int
    moves: List[Move]
    current_hp: int = field(init=False)
    max_hp: int = field(init=False)
    status: Optional[str] = None  # "BRN", "PAR", etc.

    atk: int = field(init=False)
    defense: int = field(init=False)
    spa: int = field(init=False)
    spd: int = field(init=False)
    spe: int = field(init=False)

    def __post_init__(self):
        # super-simplified stat calculation (good enough for project)
        self.max_hp = self.species.base_hp + self.level
        self.current_hp = self.max_hp

        self.atk = self.species.base_atk
        self.defense = self.species.base_def
        self.spa = self.species.base_spa
        self.spd = self.species.base_spd
        self.spe = self.species.base_spe

    def is_fainted(self) -> bool:
        return self.current_hp <= 0


@dataclass
class TrainerState:
    team: List[PokemonInstance]
    active_index: int = 0

    def active_pokemon(self) -> PokemonInstance:
        return self.team[self.active_index]

    def has_available_pokemon(self) -> bool:
        return any(not p.is_fainted() for p in self.team)

    def available_switch_indices(self) -> List[int]:
        return [i for i, p in enumerate(self.team) if i != self.active_index and not p.is_fainted()]


# =========================
# PokéAPI helpers
# =========================

def api_get(path: str) -> dict:
    url = f"{POKEAPI_BASE}{path}"
    resp = requests.get(url)
    resp.raise_for_status()
    return resp.json()


class PokemonDatabase:
    """
    Loads Gen 1-3 data from PokéAPI and provides helpers to instantiate Pokémon.

    NOTE: This hits the online API. For heavy training, you should:
      - run once
      - dump the key parts to a local JSON
      - reload from disk in later runs
    """

    # Version groups for determining legal moves across Gen 1-3
    VALID_VERSION_GROUPS = {
        # Gen 1
        "red-blue",
        "yellow",
        # Gen 2
        "gold-silver",
        "crystal",
        # Gen 3
        "ruby-sapphire",
        "emerald",
        "firered-leafgreen",
    }

    def __init__(self, generations=(1, 2, 3)):
        self.generations = generations
        self.species_map: Dict[str, PokemonSpecies] = {}
        self.move_map: Dict[str, Move] = {}
        self.type_chart: Dict[PokemonType, Dict[PokemonType, float]] = {}

    # -------- loading top-level --------
    def load_all(self):
        print(f"Loading Gen {self.generations} data from PokéAPI...")
        self._load_type_chart()
        self._load_species_and_moves()
        print(f"Loaded {len(self.species_map)} species and {len(self.move_map)} moves.")

    # -------- type chart --------
    def _load_type_chart(self):
        print("Loading type chart...")
        for t in PokemonType:
            type_name = t.value
            data = api_get(f"/type/{type_name}")
            rel = data["damage_relations"]

            # We'll store as multiplier dict: attack_type -> (def_type -> mult)
            # But here we are in "attack type" already; we want "to" multipliers.
            self.type_chart[t] = {}

            def apply(target_list, mult):
                for target in target_list:
                    def_type_name = target["name"]
                    try:
                        def_type = PokemonType(def_type_name)
                    except ValueError:
                        # ignore weird types if any
                        continue
                    self.type_chart[t][def_type] = mult * self.type_chart[t].get(def_type, 1.0)

            apply(rel["double_damage_to"], 2.0)
            apply(rel["half_damage_to"], 0.5)
            apply(rel["no_damage_to"], 0.0)

    # -------- generation species & moves --------
    def _load_species_and_moves(self):
        """Load all species and moves from specified generations."""
        for gen_id in self.generations:
            print(f"Loading Gen {gen_id} data...")
            gen_data = api_get(f"/generation/{gen_id}")
            species_entries = gen_data["pokemon_species"]
            move_entries = gen_data["moves"]

            # Load moves first so we can reuse them
            print(f"  Loading Gen {gen_id} moves...")
            for m in move_entries:
                name = m["name"]
                if name not in self.move_map:
                    mv = self._load_move(name)
                    if mv is None:
                        continue
                    self.move_map[name] = mv

            # Load species & stats
            print(f"  Loading Gen {gen_id} pokemon species...")
            for s in species_entries:
                species_name = s["name"]
                # Skip if already loaded from earlier generation
                if species_name in self.species_map:
                    continue
                    
                # The /pokemon/ endpoint uses the same names for basically all species
                species_detail = api_get(f"/pokemon-species/{species_name}")

                # Find default form (is_default = true)
                default_variety = None
                for variety in species_detail["varieties"]:
                    if variety["is_default"]:
                        default_variety = variety["pokemon"]["name"]
                        break

                if default_variety is None:
                    print(f"  Warning: No default form found for '{species_name}', skipping.")
                    continue

                poke_data = api_get(f"/pokemon/{default_variety}")
                self.species_map[species_name] = self._parse_pokemon_species(poke_data)

    def _load_move(self, name: str) -> Move:
        data = api_get(f"/move/{name}")
        move_type = PokemonType(data["type"]["name"])

        if move_type == PokemonType.SHADOW:
            return None

        damage_class_name = data["damage_class"]["name"]
        category = MoveCategory(damage_class_name)

        power = data["power"]  # can be None
        accuracy = data["accuracy"]
        if accuracy is not None:
            accuracy = accuracy / 100.0

        priority = data["priority"] or 0

        return Move(
            name=name,
            move_type=move_type,
            power=power,
            accuracy=accuracy,
            category=category,
            priority=priority,
        )

    def _parse_pokemon_species(self, data: dict) -> PokemonSpecies:
        name = data["name"]
        types = [PokemonType(t["type"]["name"]) for t in data["types"]]

        # stats are given as a list; names:
        # hp, attack, defense, special-attack, special-defense, speed
        stats = {s["stat"]["name"]: s["base_stat"] for s in data["stats"]}

        return PokemonSpecies(
            name=name,
            types=types,
            base_hp=stats["hp"],
            base_atk=stats["attack"],
            base_def=stats["defense"],
            base_spa=stats["special-attack"],
            base_spd=stats["special-defense"],
            base_spe=stats["speed"],
        )

    # -------- helpers for move selection --------

    def get_legal_moves_for_pokemon(self, pokemon_name: str) -> List[str]:
        """
        Returns move names that this Pokémon can learn in the loaded generations.
        """
        data = api_get(f"/pokemon/{pokemon_name}")
        legal_moves = []
        for m in data["moves"]:
            move_name = m["move"]["name"]
            vgroups = m["version_group_details"]
            for vgd in vgroups:
                vg_name = vgd["version_group"]["name"]
                if vg_name in self.VALID_VERSION_GROUPS:
                    legal_moves.append(move_name)
                    break
        return sorted(set(legal_moves))

    def create_pokemon_instance(
        self,
        species_name: str,
        level: int = 50,
        num_moves: int = 4,
        move_filter: Optional[str] = None,
    ) -> PokemonInstance:
        """
        Create a PokemonInstance with stats from species_map and the strongest moves
        from legal moves in the loaded generations.
        
        Moves are selected by power (highest first), then accuracy, then alphabetically.

        move_filter:
            - None: select from any legal moves
            - "special": only special moves
            - "physical": only physical moves
        """
        species_name = species_name.lower()
        if species_name not in self.species_map:
            raise ValueError(f"Unknown species '{species_name}' in Gen 3 database.")

        species = self.species_map[species_name]
        legal_move_names = self.get_legal_moves_for_pokemon(species_name)

        # Build list of Move objects, filter out weird / status moves if desired
        candidate_moves: List[Move] = []
        for name in legal_move_names:
            if name not in self.move_map:
                # Some moves might come from other generations; load on demand
                mv = self._load_move(name)
                if mv is None:
                    continue
                self.move_map[name] = mv
            mv = self.move_map[name]

            # Filter: require some power / accuracy, unless it's explicitly STATUS
            # You can tune this logic. For now, keep STATUS moves too.
            if move_filter == "physical" and mv.category != MoveCategory.PHYSICAL:
                continue
            if move_filter == "special" and mv.category != MoveCategory.SPECIAL:
                continue

            candidate_moves.append(mv)

        # Fallback: if filtering wiped everything out, use any moves
        if not candidate_moves:
            candidate_moves = [self.move_map[name] for name in legal_move_names if name in self.move_map]

        # Select the strongest moves by power
        if len(candidate_moves) <= num_moves:
            chosen = candidate_moves
        else:
            # Sort by power (descending), treating None as 0
            # Tie-break by accuracy (higher is better), then alphabetically by name
            candidate_moves.sort(
                key=lambda m: (
                    -(m.power if m.power is not None else 0),  # Highest power first
                    -(m.accuracy if m.accuracy is not None else 0),  # Then highest accuracy
                    m.name  # Then alphabetically
                )
            )
            chosen = candidate_moves[:num_moves]

        return PokemonInstance(
            species=species,
            level=level,
            moves=chosen,
        )

    # -------- type multiplier --------

    def type_multiplier(self, move_type: PokemonType, defender_types: List[PokemonType]) -> float:
        mult = 1.0
        for t in defender_types:
            mult *= self.type_chart.get(move_type, {}).get(t, 1.0)
        return mult
    
    def save_pkl(self, path: str):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load_pkl(path: str):
        with open(path, "rb") as f:
            return pickle.load(f)


# =========================
# Gen 3-style damage formula
# =========================

def calculate_damage(
    attacker: PokemonInstance,
    defender: PokemonInstance,
    move: Move,
    type_multiplier_func,
) -> Tuple[int, Dict[str, Any]]:
    """
    Simplified Gen 3 damage formula:
    damage = ((((((2 * L / 5) + 2) * P * A / D) / 50) + 2) * modifiers)
    modifiers = crit * random * stab * type_effectiveness
    """
    info = {"crit": False, "type_mult": 1.0, "stab": 1.0, "random_mult": 1.0}

    if move.power is None or move.power == 0 or move.category == MoveCategory.STATUS:
        return 0, info

    L = attacker.level
    P = move.power

    # Choose appropriate offensive/defensive stats (using modern split)
    if move.category == MoveCategory.PHYSICAL:
        A = attacker.atk
        D = defender.defense
    else:  # SPECIAL
        A = attacker.spa
        D = defender.spd

    base = math.floor(math.floor((2 * L) / 5 + 2) * P * A / D / 50) + 2

    crit = random.random() < (1.0 / 16.0)
    crit_mult = 2.0 if crit else 1.0
    info["crit"] = crit

    rand_mult = random.uniform(0.85, 1.0)
    info["random_mult"] = rand_mult

    stab = 1.5 if move.move_type in attacker.species.types else 1.0
    info["stab"] = stab

    t_mult = type_multiplier_func(move.move_type, defender.species.types)
    info["type_mult"] = t_mult

    if t_mult == 0.0:
        return 0, info

    final_damage = int(base * crit_mult * rand_mult * stab * t_mult)
    if final_damage < 1:
        final_damage = 1

    return final_damage, info


# =========================
# Battle Environment
# =========================

class ActionType(enum.Enum):
    MOVE = "MOVE"
    SWITCH = "SWITCH"


@dataclass
class ParsedAction:
    action_type: ActionType
    index: int  # move index OR switch target index


class PokemonBattleEnv:
    """
    Simple 1v1 (per side) Gen-3 style battle environment for RL.

    Interface:
      - reset() -> observation
      - step(action_self: int, action_opponent: int) -> (obs, reward, done, info)

    Actions:
      0..3: use move index
      4..(4 + num_team - 2): switch to team[index] (skipping current active)
    """

    def __init__(self, db: PokemonDatabase, trainer1: TrainerState, trainer2: TrainerState,
                 policy1: Optional['Policy'] = None, policy2: Optional['Policy'] = None):
        self.db = db
        self.initial_trainer1 = trainer1
        self.initial_trainer2 = trainer2
        self.trainer1: TrainerState = None
        self.trainer2: TrainerState = None
        self.done = False
        self.policy1 = policy1  # For forced switches
        self.policy2 = policy2  # For forced switches

    def _clone_trainer(self, t: TrainerState) -> TrainerState:
        new_team = []
        for p in t.team:
            new_p = PokemonInstance(
                species=p.species,
                level=p.level,
                moves=p.moves,
            )
            new_team.append(new_p)
        return TrainerState(team=new_team, active_index=t.active_index)

    def reset(self) -> Dict[str, Any]:
        self.trainer1 = self._clone_trainer(self.initial_trainer1)
        self.trainer2 = self._clone_trainer(self.initial_trainer2)
        self.done = False
        return self._get_observation()

    def _parse_action(self, trainer: TrainerState, action: int) -> ParsedAction:
        if action < 0:
            raise ValueError("Action must be non-negative.")
        if action <= 3:
            return ParsedAction(ActionType.MOVE, action)
        else:
            switchable = trainer.available_switch_indices()
            if not switchable:
                return ParsedAction(ActionType.MOVE, 0)
            idx = action - 4
            if idx < 0 or idx >= len(switchable):
                idx = 0
            target_team_index = switchable[idx]
            return ParsedAction(ActionType.SWITCH, target_team_index)

    def _get_observation(self) -> Dict[str, Any]:
        p1 = self.trainer1.active_pokemon()
        p2 = self.trainer2.active_pokemon()

        obs = {
            "p1_active": {
                "name": p1.species.name,
                "hp": p1.current_hp,
                "max_hp": p1.max_hp,
                "types": [t.value for t in p1.species.types],
                "atk": p1.atk,
                "def": p1.defense,
                "spa": p1.spa,
                "spd": p1.spd,
                "spe": p1.spe,
            },
            "p2_active": {
                "name": p2.species.name,
                "hp": p2.current_hp,
                "max_hp": p2.max_hp,
                "types": [t.value for t in p2.species.types],
                "atk": p2.atk,
                "def": p2.defense,
                "spa": p2.spa,
                "spd": p2.spd,
                "spe": p2.spe,
            },
            "p1_team_status": [
                {"name": p.species.name, "hp": p.current_hp, "max_hp": p.max_hp}
                for p in self.trainer1.team
            ],
            "p2_team_status": [
                {"name": p.species.name, "hp": p.current_hp, "max_hp": p.max_hp}
                for p in self.trainer2.team
            ],
        }
        return obs

    def step(self, action_self: int, action_opponent: int):
        if self.done:
            raise RuntimeError("Battle already finished. Call reset() to start a new one.")

        info = {"log": []}

        a1 = self._parse_action(self.trainer1, action_self)
        a2 = self._parse_action(self.trainer2, action_opponent)

        # Phase 1: switches
        if a1.action_type == ActionType.SWITCH:
            info["log"].append(f"Player 1 switches to {self.trainer1.team[a1.index].species.name}.")
            self.trainer1.active_index = a1.index

        if a2.action_type == ActionType.SWITCH:
            info["log"].append(f"Player 2 switches to {self.trainer2.team[a2.index].species.name}.")
            self.trainer2.active_index = a2.index

        p1 = self.trainer1.active_pokemon()
        p2 = self.trainer2.active_pokemon()

        if p1.is_fainted() or p2.is_fainted():
            self._handle_faints(info)
            obs = self._get_observation()
            reward, done = self._compute_reward_done()
            self.done = done
            return obs, reward, done, info

        # Collect attacks
        attack_actions: List[Tuple[int, TrainerState, ParsedAction]] = []
        if a1.action_type == ActionType.MOVE:
            attack_actions.append((1, self.trainer1, a1))
        if a2.action_type == ActionType.MOVE:
            attack_actions.append((2, self.trainer2, a2))

        ordered_attacks = []
        for player_id, trainer, pa in attack_actions:
            active = trainer.active_pokemon()
            move_idx = pa.index
            if move_idx >= len(active.moves):
                move_idx = 0
            move = active.moves[move_idx]
            priority = move.priority or 0
            speed = active.spe
            ordered_attacks.append((priority, speed, player_id, trainer, pa, move))
        ordered_attacks.sort(key=lambda x: (-x[0], -x[1], random.random()))

        # Execute attacks
        for priority, speed, player_id, trainer, pa, move in ordered_attacks:
            if self.done:
                break
            if not trainer.active_pokemon().is_fainted():
                self._execute_move(player_id, move, info)
            self._handle_faints(info)

        reward, done = self._compute_reward_done()
        self.done = done
        obs = self._get_observation()
        return obs, reward, done, info

    def _execute_move(self, player_id: int, move: Move, info: Dict[str, Any]):
        if player_id == 1:
            attacker_trainer = self.trainer1
            defender_trainer = self.trainer2
        else:
            attacker_trainer = self.trainer2
            defender_trainer = self.trainer1

        attacker = attacker_trainer.active_pokemon()
        defender = defender_trainer.active_pokemon()

        if attacker.is_fainted():
            return

        # Accuracy
        if move.accuracy is not None and random.random() > move.accuracy:
            info["log"].append(f"Player {player_id}'s {attacker.species.name} used {move.name}, but it missed!")
            return

        damage, dmg_info = calculate_damage(attacker, defender, move, self.db.type_multiplier)

        defender.current_hp -= damage
        if defender.current_hp < 0:
            defender.current_hp = 0

        msg = (
            f"Player {player_id}'s {attacker.species.name} used {move.name} "
            f"and dealt {damage} damage to {defender.species.name} "
            f"(HP: {defender.current_hp}/{defender.max_hp})."
        )
        if dmg_info["crit"]:
            msg += " A critical hit!"
        if dmg_info["type_mult"] > 1.0:
            msg += " It's super effective!"
        elif 0 < dmg_info["type_mult"] < 1.0:
            msg += " It's not very effective..."
        elif dmg_info["type_mult"] == 0.0:
            msg += " It had no effect."

        info["log"].append(msg)

    def _handle_faints(self, info: Dict[str, Any]):
        for player_id, trainer in [(1, self.trainer1), (2, self.trainer2)]:
            active = trainer.active_pokemon()
            if active.is_fainted():
                info["log"].append(
                    f"Player {player_id}'s {active.species.name} fainted!"
                )
                if trainer.has_available_pokemon():
                    available = trainer.available_switch_indices()
                    
                    # Use policy to select which Pokemon to switch to
                    policy = self.policy1 if player_id == 1 else self.policy2
                    if policy is not None:
                        obs = self._get_observation()
                        # Adjust observation for player 2 (swap p1/p2 in obs)
                        if player_id == 2:
                            obs = {
                                "p1_active": obs["p2_active"],
                                "p2_active": obs["p1_active"],
                                "p1_team_status": obs["p2_team_status"],
                                "p2_team_status": obs["p1_team_status"],
                            }
                        switch_to = policy.select_forced_switch(obs, available)
                    else:
                        # Default: pick first available
                        switch_to = available[0]
                    
                    trainer.active_index = switch_to
                    info["log"].append(
                        f"Player {player_id} sends out {trainer.team[switch_to].species.name}!"
                    )

    def _compute_reward_done(self) -> Tuple[float, bool]:
        p1_alive = self.trainer1.has_available_pokemon()
        p2_alive = self.trainer2.has_available_pokemon()
        if p1_alive and p2_alive:
            # shaping: hp diff
            p1_hp = sum(p.current_hp for p in self.trainer1.team)
            p2_hp = sum(p.current_hp for p in self.trainer2.team)
            reward = (p1_hp - p2_hp) / 100.0
            return reward, False
        elif p1_alive and not p2_alive:
            return 1.0, True
        elif not p1_alive and p2_alive:
            return -1.0, True
        else:
            return 0.0, True


# =========================
# Policy Interface
# =========================

class Policy:
    """Base class for battle policies."""
    
    def select_action(self, obs: Dict[str, Any], valid_moves: List[Move], 
                     available_switches: List[int]) -> int:
        """
        Select an action based on observation.
        
        Args:
            obs: Current battle observation with team status
            valid_moves: List of available moves for the active Pokemon
            available_switches: List of team indices that can be switched to
            
        Returns:
            Action index:
            - 0-3: use move at that index
            - 4+: switch to Pokemon (action_index - 4 maps to available_switches)
        """
        raise NotImplementedError
    
    def select_forced_switch(self, obs: Dict[str, Any], 
                            available_switches: List[int]) -> int:
        """
        Select which Pokemon to switch to when current one faints.
        
        Args:
            obs: Current battle observation
            available_switches: List of team indices that can be switched to
            
        Returns:
            Team index to switch to
        """
        # Default: pick first available
        return available_switches[0] if available_switches else 0


class RandomPolicy(Policy):
    """Randomly selects from available moves and switches."""
    
    def __init__(self, switch_probability: float = 0.1):
        """
        Args:
            switch_probability: Probability of switching instead of attacking
        """
        self.switch_probability = switch_probability
    
    def select_action(self, obs: Dict[str, Any], valid_moves: List[Move], 
                     available_switches: List[int]) -> int:
        # Randomly decide whether to switch
        if available_switches and random.random() < self.switch_probability:
            # Switch action: 4 + random index into available_switches
            switch_idx = random.randint(0, len(available_switches) - 1)
            return 4 + switch_idx
        else:
            # Move action
            return random.randint(0, min(3, len(valid_moves) - 1))
    
    def select_forced_switch(self, obs: Dict[str, Any], 
                            available_switches: List[int]) -> int:
        # Randomly pick from available Pokemon
        return random.choice(available_switches) if available_switches else 0


class GreedyPowerPolicy(Policy):
    """Selects the move with highest power (doesn't switch voluntarily)."""
    
    def select_action(self, obs: Dict[str, Any], valid_moves: List[Move],
                     available_switches: List[int]) -> int:
        best_idx = 0
        best_power = -1
        
        for i, move in enumerate(valid_moves):
            power = move.power if move.power is not None else 0
            if power > best_power:
                best_power = power
                best_idx = i
        
        return best_idx
    
    def select_forced_switch(self, obs: Dict[str, Any], 
                            available_switches: List[int]) -> int:
        # Pick Pokemon with highest HP
        team_status = obs["p1_team_status"]
        best_idx = available_switches[0]
        best_hp = 0
        
        for idx in available_switches:
            hp = team_status[idx]["hp"]
            if hp > best_hp:
                best_hp = hp
                best_idx = idx
        
        return best_idx


class GreedyTypePolicy(Policy):
    """Selects move with best type advantage; switches if heavily disadvantaged."""
    
    def __init__(self, db: PokemonDatabase, switch_threshold: float = 0.3):
        """
        Args:
            db: Pokemon database for type calculations
            switch_threshold: Switch if best move effectiveness is below this
        """
        self.db = db
        self.switch_threshold = switch_threshold
    
    def select_action(self, obs: Dict[str, Any], valid_moves: List[Move],
                     available_switches: List[int]) -> int:
        # Get opponent types
        opp_types = [PokemonType(t) for t in obs["p2_active"]["types"]]
        
        best_idx = 0
        best_score = -1
        
        for i, move in enumerate(valid_moves):
            if move.power is None or move.power == 0:
                score = 0
            else:
                type_mult = self.db.type_multiplier(move.move_type, opp_types)
                score = move.power * type_mult
            
            if score > best_score:
                best_score = score
                best_idx = i
        
        # Consider switching if our best move is weak
        if available_switches and best_score < self.switch_threshold * 100:
            # Check if we have a better Pokemon to switch to
            my_hp_percent = obs["p1_active"]["hp"] / obs["p1_active"]["max_hp"]
            
            # If we're low HP and our moves are weak, switch
            if my_hp_percent < 0.5:
                return 4 + random.randint(0, len(available_switches) - 1)
        
        return best_idx
    
    def select_forced_switch(self, obs: Dict[str, Any], 
                            available_switches: List[int]) -> int:
        # Pick Pokemon with best type matchup (or highest HP as fallback)
        team_status = obs["p1_team_status"]
        best_idx = available_switches[0]
        best_hp = team_status[best_idx]["hp"]
        
        for idx in available_switches:
            hp = team_status[idx]["hp"]
            if hp > best_hp:
                best_hp = hp
                best_idx = idx
        
        return best_idx


# =========================
# Example: Starter Battle using real data
# =========================

def make_starter_battle_env(db: PokemonDatabase, level: int = 10, 
                           team_size: int = 3) -> PokemonBattleEnv:
    """
    Example battle environment with teams of Pokemon.

    You can use any Gen 1-3 Pokémon, e.g.:
    - Gen 1: pikachu, charizard, mewtwo
    - Gen 2: typhlosion, feraligatr, lugia
    - Gen 3: torchic, treecko, rayquaza
    
    Args:
        db: Pokemon database
        level: Level for all Pokemon
        team_size: Number of Pokemon per team (1-6)
    """
    # Example teams - you can customize these!
    if team_size == 1:
        team1 = [db.create_pokemon_instance("torchic", level=level)]
        team2 = [db.create_pokemon_instance("treecko", level=level)]
    else:
        # Create diverse teams from different generations
        team1_species = ["torchic", "pikachu", "cyndaquil", "squirtle", "mudkip", "totodile"]
        team2_species = ["treecko", "charmander", "chikorita", "bulbasaur", "marshtomp", "croconaw"]
        
        team1 = [db.create_pokemon_instance(species, level=level) 
                for species in team1_species[:team_size]]
        team2 = [db.create_pokemon_instance(species, level=level) 
                for species in team2_species[:team_size]]

    t1 = TrainerState(team=team1, active_index=0)
    t2 = TrainerState(team=team2, active_index=0)

    env = PokemonBattleEnv(db=db, trainer1=t1, trainer2=t2)
    return env


# =========================
# Manual test
# =========================

if __name__ == "__main__":
    random.seed(67)

    # Load all Gen 1-3 Pokemon and moves
    db = PokemonDatabase(generations=(1, 2, 3))
    db.load_all()

    # Example: Create battle with teams of Pokemon from different generations
    # team_size=3 creates teams of 3 Pokemon each
    env = make_starter_battle_env(db, level=10, team_size=3)
    obs = env.reset()
    
    # Initialize policies - try different ones!
    # policy1 = RandomPolicy(switch_probability=0.2)  # Switches 20% of the time
    # policy1 = GreedyPowerPolicy()
    policy1 = GreedyTypePolicy(db)
    policy2 = RandomPolicy(switch_probability=0.15)
    
    print("=== Battle Start ===")
    print("\nPlayer 1 Team:")
    for i, p in enumerate(env.trainer1.team):
        moves_str = ", ".join([m.name for m in p.moves])
        print(f"  {i}. {p.species.name} (HP: {p.max_hp}) - Moves: {moves_str}")
    
    print("\nPlayer 2 Team:")
    for i, p in enumerate(env.trainer2.team):
        moves_str = ", ".join([m.name for m in p.moves])
        print(f"  {i}. {p.species.name} (HP: {p.max_hp}) - Moves: {moves_str}")
    
    print(f"\nStarting: {obs['p1_active']['name']} vs {obs['p2_active']['name']}")
    print()

    # Set policies in environment for forced switch handling
    env.policy1 = policy1
    env.policy2 = policy2
    
    done = False
    step_count = 0
    while not done and step_count < 50:
        # Policies select actions based on observation
        # Include available switches so policy can choose to switch
        available_switches_p1 = env.trainer1.available_switch_indices()
        available_switches_p2 = env.trainer2.available_switch_indices()
        
        a_self = policy1.select_action(
            obs, 
            env.trainer1.active_pokemon().moves,
            available_switches_p1
        )
        
        # For player 2, swap observations so they see themselves as p1
        obs_p2 = {
            "p1_active": obs["p2_active"],
            "p2_active": obs["p1_active"],
            "p1_team_status": obs["p2_team_status"],
            "p2_team_status": obs["p1_team_status"],
        }
        a_opp = policy2.select_action(
            obs_p2,
            env.trainer2.active_pokemon().moves,
            available_switches_p2
        )
        
        obs, reward, done, info = env.step(a_self, a_opp)

        print(f"\n--- Turn {step_count + 1} ---")
        for line in info["log"]:
            print(line)
        print("Reward:", reward, "Done:", done)

        step_count += 1