from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Callable, List, Tuple


# ---------------- I/O JSON ----------------

def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, obj: Any, *, indent: int = 2) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=indent)


# ---------------- Logica statistica ----------------

def action_l2_distance(a1: Tuple[float, float], a2: Tuple[float, float]) -> float:
    """Distanza euclidea tra due azioni (a1,a2) 2D."""
    return math.hypot(a2[0] - a1[0], a2[1] - a1[1])


def mean_successive_action_diff_for_episode(
    episode_transitions: List[List[Any]],
    *,
    distance_fn: Callable[[Tuple[float, float], Tuple[float, float]], float] = action_l2_distance,
    on_short_episode: float = 0.0,
) -> float:
    """
    episode_transitions = [t1, t2, ...]
    t = [..., a1, a2]  (azioni negli ultimi 2 posti)

    Ritorna la media delle distanze tra azioni successive.
    Se l'episodio ha <2 transizioni, ritorna on_short_episode.
    """
    actions: List[Tuple[float, float]] = []
    for idx, t in enumerate(episode_transitions):
        if not isinstance(t, list) or len(t) < 2:
            raise ValueError(f"Transizione {idx} non valida: attesa lista lunga almeno 2.")
        try:
            a = (float(t[-2]), float(t[-1]))
        except Exception as e:
            raise ValueError(f"Transizione {idx}: ultimi 2 valori non convertibili in float: {t[-2:]}") from e
        actions.append(a)

    if len(actions) < 2:
        return on_short_episode

    diffs = [distance_fn(actions[i], actions[i + 1]) for i in range(len(actions) - 1)]
    return sum(diffs) / len(diffs)


def mean_successive_action_diff_all_episodes(
    transitions_obj: Any,
    *,
    distance_fn: Callable[[Tuple[float, float], Tuple[float, float]], float] = action_l2_distance,
    on_short_episode: float = 0.0,
) -> List[float]:
    """
    transitions_obj = [episode1, episode2, ...]
    episode = [t1, t2, ...]
    t = [..., a1, a2]
    """
    if not isinstance(transitions_obj, list):
        raise ValueError("Il file _transitions.json deve essere una lista di episodi.")

    out: List[float] = []
    for ep_idx, ep in enumerate(transitions_obj):
        if not isinstance(ep, list):
            raise ValueError(f"Episodio {ep_idx} non valido: attesa lista di transizioni.")
        out.append(
            mean_successive_action_diff_for_episode(
                ep,
                distance_fn=distance_fn,
                on_short_episode=on_short_episode,
            )
        )
    return out


# ---------------- Controllo/Update info ----------------

def info_needs_stat(info_obj: Any, *, stat_key: str) -> bool:
    """
    True se info_obj ha forma {'metadata':..., 'data':[dict,...]}
    e almeno un dict in data non ha stat_key.
    """
    if not isinstance(info_obj, dict):
        return False
    data = info_obj.get("data")
    if not isinstance(data, list) or not data:
        return False
    if not all(isinstance(e, dict) for e in data):
        return False
    return any(stat_key not in e for e in data)


def update_one_info_file(
    info_path: Path,
    save_path: Path,
    *,
    stat_key: str = "mean_successive_action_diff",
    json_indent: int = 2,
    distance_fn: Callable[[Tuple[float, float], Tuple[float, float]], float] = action_l2_distance,
    on_short_episode: float = 0.0,
) -> bool:
    """
    Aggiorna un singolo *_info.json.
    Cerca il corrispondente *_transitions.json (stesso prefisso).
    Ritorna True se ha scritto modifiche.
    """
    info_obj = read_json(info_path)
    if not info_needs_stat(info_obj, stat_key=stat_key):
        return False

    data: List[dict] = info_obj["data"]

    transitions_path = Path(str(info_path).replace("_info.json", "_transitions.json"))
    if not transitions_path.exists():
        raise FileNotFoundError(f"Manca il file atteso: {transitions_path}")

    transitions_obj = read_json(transitions_path)
    values = mean_successive_action_diff_all_episodes(
        transitions_obj,
        distance_fn=distance_fn,
        on_short_episode=on_short_episode,
    )

    if len(values) != len(data):
        raise ValueError(
            f"Mismatch episodi: {info_path.name} ha {len(data)} episodi, "
            f"{transitions_path.name} ne ha {len(values)}."
        )

    changed = False
    for i, ep_stat in enumerate(data):
        if stat_key not in ep_stat:
            ep_stat[stat_key] = values[i]
            changed = True

    if not changed:
        return False

    write_json(save_path, info_obj, indent=json_indent)
    return True


def update_folder_recursively(
    root_dir: str | Path,
    save_dir: str | Path = None,
    *,
    stat_key: str = "action_difference",
    json_indent: int = 2,
    distance_fn: Callable[[Tuple[float, float], Tuple[float, float]], float] = action_l2_distance,
    on_short_episode: float = 0.0) -> List[Path]:
    """
    Scansiona ricorsivamente root_dir, aggiorna tutti i *_info.json.
    Ritorna lista dei file aggiornati.
    """
    root = Path(root_dir)
    save_root = Path(save_dir) if save_dir is not None else None

    updated: List[Path] = []

    for info_path in root.rglob("*_info.json"):
        try:
            # Costruzione save_path con suffisso _ad
            if save_root is not None:
                rel_path = info_path.relative_to(root)
                target_base = save_root / rel_path
            else:
                target_base = info_path

            save_path = target_base.with_name(
                f"{target_base.stem}_ad{target_base.suffix}"
            )

            save_path.parent.mkdir(parents=True, exist_ok=True)

            if update_one_info_file(
                info_path,
                save_path=save_path,
                stat_key=stat_key,
                json_indent=json_indent,
                distance_fn=distance_fn,
                on_short_episode=on_short_episode,
            ):
                updated.append(save_path)

        except Exception as e:
            print(f"[WARN] {info_path}: {e}")

    return updated

# ---------------- Esempio ----------------
updated = update_folder_recursively("./test_ad", "./test_ad")
print("Aggiornati:", [str(p) for p in updated])