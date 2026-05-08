"""Prepare SocialMaze train/test splits for SkillGen (FTS / RDP / UPI).

Three sub-commands (one per task):

    python prepare_socialmaze.py rdp --train-n 24 --test-n 16 --seed 42 \
        --out-dir data/socialmaze/rdp

    python prepare_socialmaze.py fts --pool-size 120 --train-n 60 --test-n 50 \
        --gen-model openai/gpt-5.4-nano --seed 42 --out-dir data/socialmaze/fts

    python prepare_socialmaze.py upi --pool-size 120 --train-n 60 --test-n 50 \
        --variant persona --gen-model openai/gpt-5.4-nano --seed 42 \
        --out-dir data/socialmaze/upi

RDP uses the shipped 40-item debate.json (no generation needed). FTS and
UPI will generate a pool via OpenRouter on first run (cached as
`<out-dir>/_pool.json`) then slice disjoint train/test windows.

Everything written here is compatible with `main.py` / `eval_skill.py`.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import random
import re
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from benchmarks.socialmaze_adapter import (
    convert_fts_item,
    convert_rdp_item,
    convert_upi_entity_item,
    convert_upi_persona_item,
    load_fts_shipped,
    load_rdp_shipped,
    load_upi_entity_shipped,
    load_upi_persona_shipped,
)

import llm
_SOCIALMAZE_ROOT = _REPO_ROOT / "external" / "social-maze"


# Dataset writer (shared)

def _write_dataset(
    *,
    instances: list[dict[str, Any]],
    task: str,
    split_label: str,
    seed: int,
    pool_total: int,
    out_path: Path,
    extra_meta: dict[str, Any] | None = None,
) -> None:
    # UPI returns partial scores (0.5 when only age or gender is correct),
    # so it maps to the "scored" TaskType; FTS and RDP are clean 0/1.
    task_type_map = {"fts": "binary", "rdp": "binary", "upi": "scored"}
    payload = {
        "dataset_id": f"socialmaze_{task}_{split_label}_n{len(instances)}_seed{seed}",
        "task_name": f"socialmaze_{task}_{split_label}",
        "task_type": task_type_map.get(task, "open_ended"),
        "instances": instances,
        "metadata": {
            "benchmark": "socialmaze",
            "task": task,
            "split_label": split_label,
            "n": len(instances),
            "seed": seed,
            "pool_total": pool_total,
            "source": "xzx34/SocialMaze",
            **(extra_meta or {}),
        },
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"  Wrote {len(instances)} instances -> {out_path}")


def _split_pool(
    pool: list[dict[str, Any]],
    *,
    train_n: int,
    test_n: int,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if train_n + test_n > len(pool):
        raise SystemExit(
            f"Pool only has {len(pool)} items but train_n={train_n} + test_n={test_n} "
            f"= {train_n + test_n} were requested. Regenerate with a larger "
            "--pool-size, or shrink your splits."
        )
    rng = random.Random(seed)
    shuf = list(pool)
    rng.shuffle(shuf)
    train = shuf[:train_n]
    test = shuf[train_n : train_n + test_n]

    train_ids = {t["instance_id"] for t in train}
    test_ids = {t["instance_id"] for t in test}
    overlap = train_ids & test_ids
    if overlap:
        raise RuntimeError(f"unexpected train/test overlap: {sorted(overlap)[:5]}")
    return train, test


# RDP


def prepare_rdp(args: argparse.Namespace) -> None:
    raw = load_rdp_shipped(args.data_path)
    if not raw:
        raise SystemExit(
            "Could not load external/social-maze/review_decision_prediction/"
            "data/debate.json. Is the submodule present?"
        )

    instances = []
    for idx, item in enumerate(raw):
        inst = convert_rdp_item(item, idx=idx)
        if inst is not None:
            instances.append(inst)

    print(f"Loaded {len(raw)} raw RDP items, {len(instances)} usable after "
          "filtering unknown decisions.")

    train, test = _split_pool(
        instances, train_n=args.train_n, test_n=args.test_n, seed=args.seed
    )
    out_dir = Path(args.out_dir)
    _write_dataset(
        instances=train, task="rdp", split_label="train", seed=args.seed,
        pool_total=len(instances),
        out_path=out_dir / f"train_n{len(train)}_seed{args.seed}.json",
    )
    _write_dataset(
        instances=test, task="rdp", split_label="test", seed=args.seed,
        pool_total=len(instances),
        out_path=out_dir / f"test_n{len(test)}_seed{args.seed}.json",
    )


# FTS pool generation

def _load_fts_word_pairs() -> list[tuple[str, str]]:
    """Import WORD_PAIRS from the submodule by file path (avoids sys.path shuffling)."""
    word_pairs_path = _SOCIALMAZE_ROOT / "find_the_spy" / "word_pairs.py"
    if not word_pairs_path.exists():
        raise SystemExit(f"Could not find {word_pairs_path}")
    spec = importlib.util.spec_from_file_location("_fts_word_pairs", word_pairs_path)
    mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    assert spec and spec.loader
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return [tuple(p) for p in getattr(mod, "WORD_PAIRS")]


def _fts_player_gen_prompt(word: str, player_id: str, num_rounds: int) -> tuple[str, str]:
    """Return (system, user) prompt for generating one player's N-round descriptions."""
    rounds_fmt = "\n".join(
        f"Round {i + 1} description: [your description from a different angle]"
        for i in range(num_rounds)
    )
    system = (
        "You are helping a player describe a word in a word-guessing game. "
        "Produce ONLY the requested description lines, nothing else."
    )
    user = (
        f"You are playing a word description game with 4 players including yourself. "
        f"You are Player {player_id}.\n\n"
        "Game Rules:\n"
        "1. In this game, 3 players receive the same word (normal word), and 1 "
        "player receives a different but related word (spy word).\n"
        "2. Each player must describe their word without saying the actual word.\n"
        "3. If you're a normal player, your goal is to identify the spy.\n"
        "4. If you're the spy, your goal is to blend in and avoid detection.\n\n"
        f'Your word is: "{word}"\n\n'
        "Format your response as:\n"
        f"{rounds_fmt}\n\n"
        "Remember:\n"
        "- Don't use the actual word in your descriptions\n"
        "- In each round, describe your word from a different perspective or "
        "focus on different aspects\n"
        "- Balance between being clear enough for teammates but not too obvious\n"
        "- Each description should be 1 sentence only"
    )
    return system, user


_FTS_ROUND_RE = re.compile(r"Round\s+(\d+)\s+description\s*:\s*(.*)", re.IGNORECASE)


def _parse_fts_descriptions(raw: str, num_rounds: int) -> dict[str, str]:
    """Parse 'Round 1 description: ...' lines out of the model response."""
    rounds: dict[str, str] = {}
    # Match each round by scanning until the next "Round N description:"
    for i in range(1, num_rounds + 1):
        if i < num_rounds:
            patt = re.compile(
                rf"Round\s+{i}\s+description\s*:\s*(.*?)(?=Round\s+{i + 1}\s+description|$)",
                re.IGNORECASE | re.DOTALL,
            )
        else:
            patt = re.compile(
                rf"Round\s+{i}\s+description\s*:\s*(.*?)$",
                re.IGNORECASE | re.DOTALL,
            )
        m = patt.search(raw)
        rounds[f"round{i}"] = m.group(1).strip() if m else "No description provided."
    return rounds


def _fts_generate_scenario(
    scenario_id: str,
    word_pairs: list[tuple[str, str]],
    force_spy_is_p1: bool | None,
    num_rounds: int,
    gen_model: str,
    rng: random.Random,
) -> dict[str, Any] | None:
    normal_word, spy_word = rng.choice(word_pairs)
    if force_spy_is_p1 is True:
        spy_player = "1"
    elif force_spy_is_p1 is False:
        spy_player = str(rng.randint(2, 4))
    else:
        spy_player = str(rng.randint(1, 4))

    player_words = {
        str(i): (spy_word if str(i) == spy_player else normal_word) for i in range(1, 5)
    }

    descriptions: dict[str, dict[str, str]] = {}
    for pid in ("1", "2", "3", "4"):
        system, user = _fts_player_gen_prompt(player_words[pid], pid, num_rounds)
        try:
            response = llm.chat(user, system=system, model=gen_model, temperature=0.7,
                                max_tokens=400)
        except Exception as exc:
            print(f"  [{scenario_id}] player {pid} gen failed: {exc}")
            return None
        descriptions[pid] = _parse_fts_descriptions(str(response or ""), num_rounds)

    statements = []
    for r in range(1, num_rounds + 1):
        round_block = {"round": r, "statements": []}
        for pid in ("1", "2", "3", "4"):
            text = descriptions[pid].get(f"round{r}", "")
            round_block["statements"].append(
                {"player": pid, "statement": f"Player {pid}: {text}"}
            )
        statements.append(round_block)

    return {
        "scenario_id": scenario_id,
        "normal_word": normal_word,
        "spy_word": spy_word,
        "spy_player": spy_player,
        "player_words": player_words,
        "num_rounds": num_rounds,
        "descriptions": descriptions,
        "statements": statements,
    }


def _fts_generate_pool(
    pool_size: int,
    num_rounds: int,
    gen_model: str,
    seed: int,
    max_workers: int,
) -> list[dict[str, Any]]:
    word_pairs = _load_fts_word_pairs()
    rng = random.Random(seed)

    # Keep P1-is-spy vs P1-not-spy balanced.
    jobs: list[tuple[str, bool]] = []
    spy_count = pool_size // 2
    for i in range(1, spy_count + 1):
        jobs.append((f"spy_{i}", True))
    for i in range(1, pool_size - spy_count + 1):
        jobs.append((f"normal_{i}", False))
    rng.shuffle(jobs)

    def _run(idx: int, scenario_id: str, force_spy: bool) -> dict[str, Any] | None:
        # Use a per-scenario RNG so parallel calls stay deterministic.
        local_rng = random.Random(seed + idx)
        return _fts_generate_scenario(
            scenario_id=scenario_id,
            word_pairs=word_pairs,
            force_spy_is_p1=force_spy,
            num_rounds=num_rounds,
            gen_model=gen_model,
            rng=local_rng,
        )

    with llm.stage_scope("socialmaze_fts_gen"):
        args_list = [(i, sid, spy) for i, (sid, spy) in enumerate(jobs)]
        results = llm.run_concurrent(
            _run, args_list, max_workers=max_workers,
            progress_desc="FTS scenarios",
        )

    pool = [r for r in results if r is not None]
    print(f"FTS: generated {len(pool)}/{pool_size} scenarios "
          f"({pool_size - len(pool)} failed).")
    return pool


def prepare_fts(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pool_path = out_dir / f"_pool_n{args.pool_size}_rounds{args.num_rounds}_seed{args.seed}.json"

    if pool_path.exists() and not args.regen:
        pool = json.loads(pool_path.read_text())
        print(f"FTS: using cached pool {pool_path} ({len(pool)} scenarios).")
    else:
        # Seed pool with any shipped scenarios first (they're free!)
        shipped = load_fts_shipped()
        target_extra = max(0, args.pool_size - len(shipped))
        print(f"FTS: {len(shipped)} shipped scenarios; generating {target_extra} more "
              f"via {args.gen_model}...")
        generated = _fts_generate_pool(
            pool_size=target_extra,
            num_rounds=args.num_rounds,
            gen_model=args.gen_model,
            seed=args.seed,
            max_workers=args.max_workers,
        )
        # Prefix shipped scenario_ids so they don't collide with generated ones
        for i, s in enumerate(shipped):
            s.setdefault("scenario_id", f"shipped_{i}")
            s["scenario_id"] = f"shipped_{s['scenario_id']}"
            s.setdefault("num_rounds", args.num_rounds)
        pool = shipped + generated
        pool_path.write_text(json.dumps(pool, ensure_ascii=False, indent=2))
        print(f"FTS: cached pool -> {pool_path} ({len(pool)} scenarios).")

    instances = [convert_fts_item(s, idx=i) for i, s in enumerate(pool)]

    train, test = _split_pool(
        instances, train_n=args.train_n, test_n=args.test_n, seed=args.seed
    )
    _write_dataset(
        instances=train, task="fts", split_label="train", seed=args.seed,
        pool_total=len(instances),
        out_path=out_dir / f"train_n{len(train)}_seed{args.seed}.json",
        extra_meta={"num_rounds": args.num_rounds, "gen_model": args.gen_model},
    )
    _write_dataset(
        instances=test, task="fts", split_label="test", seed=args.seed,
        pool_total=len(instances),
        out_path=out_dir / f"test_n{len(test)}_seed{args.seed}.json",
        extra_meta={"num_rounds": args.num_rounds, "gen_model": args.gen_model},
    )


# UPI pool generation

_UPI_AGE_GROUPS = ("18-34", "35-54", "55+")
_UPI_GENDERS = ("Male", "Female", "Non-binary")


_UPI_CONTENT_SUBJECTS = [
    # Electronics
    "iPhone 15 Pro Max", "Samsung Galaxy S24", "Google Pixel 8 Pro", "OnePlus 12",
    "iPad Pro 2023", "Samsung Galaxy Tab S9", "Amazon Fire HD 10", "Lenovo Tab P12 Pro",
    "MacBook Air M2", "Dell XPS 13", "HP Spectre x360", "Lenovo ThinkPad X1 Carbon",
    "Sony WH-1000XM5 Headphones", "Bose QuietComfort Earbuds", "Apple AirPods Pro 2", "Sennheiser Momentum 4",
    "Nintendo Switch OLED", "PlayStation 5", "Xbox Series X", "Steam Deck",
    "Kindle Paperwhite", "Kobo Libra 2", "reMarkable 2 Tablet", "Onyx Boox Note Air 2",
    # Home Appliances
    "Dyson V12 Vacuum", "Roomba j7+", "Shark Navigator Lift-Away", "Miele Complete C3",
    "Instant Pot Duo Crisp", "Ninja Foodi", "KitchenAid Stand Mixer", "Vitamix E310 Blender",
    "Samsung Family Hub Refrigerator", "LG InstaView Refrigerator", "Whirlpool Top-Freezer", "GE Profile Smart Refrigerator",
    "IKEA HEMNES Dresser", "West Elm Mid-Century Bed", "La-Z-Boy Recliner", "Herman Miller Aeron Chair",
    # Transportation
    "Tesla Model Y", "Toyota RAV4 Hybrid", "Honda Civic", "Ford F-150 Lightning",
    "VanMoof S5 E-Bike", "Rad Power RadRunner", "Brompton Folding Bike", "Trek Domane SL6",
    # Clothing
    "Nike Air Zoom Pegasus 39", "Adidas Ultraboost 23", "Hoka Clifton 9", "New Balance 990v6",
    "Lululemon Align Leggings", "Nike Dri-FIT Running Shorts", "Patagonia Better Sweater", "The North Face Thermoball Jacket",
    "Apple Watch Series 9", "Garmin Fenix 7", "Fitbit Charge 6", "Samsung Galaxy Watch 6",
    # Software / Services
    "ChatGPT Plus Subscription", "Microsoft 365 Family Plan", "Adobe Creative Cloud", "Notion Premium",
    "Netflix Premium Plan", "Disney+ Bundle", "Spotify Premium", "YouTube Premium",
    "Amazon Prime Membership", "Costco Gold Star Membership", "Sam's Club Plus", "Walmart+ Membership",
    # Entertainment
    "The Last of Us TV Show", "Succession", "House of the Dragon", "Ted Lasso",
    "Oppenheimer Movie", "Barbie Movie", "Dune: Part Two", "Poor Things",
    "Call of Duty: Modern Warfare 3", "Baldur's Gate 3",
    "The Legend of Zelda: Tears of the Kingdom", "Starfield",
    # Food & Drinks
    "Starbucks Pumpkin Spice Latte", "McDonald's McSpicy", "Chipotle Burrito Bowl", "Shake Shack ShackBurger",
    "Coca-Cola Zero Sugar", "LaCroix Sparkling Water", "Liquid Death Mountain Water", "Athletic Brewing Non-Alcoholic Beer",
    # Health & Beauty
    "Dyson Airwrap", "Theragun Elite", "Olaplex Hair Treatment", "Cerave Moisturizing Cream",
    "Peloton Bike+", "Hydrow Rowing Machine", "WHOOP 4.0 Fitness Tracker", "Theragun Mini",
    # Travel
    "Airbnb Plus Stays", "Marriott Bonvoy Program", "TSA PreCheck Membership", "Away The Carry-On Suitcase",
    # Books
    "Atomic Habits by James Clear", "Fourth Wing by Rebecca Yarros",
    "The Housemaid by Freida McFadden", "Iron Flame by Rebecca Yarros",
]


def _upi_make_personas(
    num_personas: int, rng: random.Random
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    """Create `num_personas` personas with a clear majority age+gender."""
    primary_age_idx = rng.randint(0, 1)  # 18-34 or 35-54 (match original upi_gen)
    primary_age = _UPI_AGE_GROUPS[primary_age_idx]
    primary_gender = rng.choice(("Male", "Female"))

    # ~80% primary, ~20% split among others
    personas: list[dict[str, Any]] = []
    for i in range(num_personas):
        age_group = (
            primary_age if rng.random() < 0.8
            else rng.choice([a for a in _UPI_AGE_GROUPS if a != primary_age])
        )
        gender = (
            primary_gender if rng.random() < 0.8
            else rng.choice([g for g in _UPI_GENDERS if g != primary_gender])
        )
        personas.append({"id": i + 1, "age_group": age_group, "gender": gender})

    # Enforce majorities (at least 70% age, 60% gender)
    from collections import Counter
    age_counts = Counter(p["age_group"] for p in personas)
    gender_counts = Counter(p["gender"] for p in personas)
    if age_counts[primary_age] / num_personas < 0.7 or gender_counts[primary_gender] / num_personas < 0.6:
        # Force majority deterministically
        n_age_majority = max(int(num_personas * 0.75), 1)
        n_gender_majority = max(int(num_personas * 0.7), 1)
        for i, p in enumerate(personas):
            p["age_group"] = primary_age if i < n_age_majority else rng.choice(
                [a for a in _UPI_AGE_GROUPS if a != primary_age]
            )
            p["gender"] = primary_gender if i < n_gender_majority else rng.choice(
                [g for g in _UPI_GENDERS if g != primary_gender]
            )

    return personas, {
        "primary_age_group": primary_age,
        "primary_gender": primary_gender,
    }


def _upi_comment_prompt(persona: dict[str, Any], subject: str) -> tuple[str, str]:
    age = persona["age_group"]
    gender = persona["gender"]
    system = (
        f"You are a {age} {gender} user providing a comment about this product: "
        f'"{subject}". Write a realistic, authentic review (2-4 sentences) '
        "that sounds like a genuine user, not a marketing expert or professional "
        "reviewer, and that may reflect common writing patterns of your age group. "
        "Return ONLY the comment text, with no framing or quotation marks."
    )
    user = f"Please comment on this product: {subject}"
    return system, user


def _upi_gen_one_comment(
    persona: dict[str, Any], subject: str, gen_model: str
) -> dict[str, Any] | None:
    system, user = _upi_comment_prompt(persona, subject)
    try:
        resp = llm.chat(user, system=system, model=gen_model,
                        temperature=1.0, max_tokens=200)
    except Exception as exc:
        print(f"  UPI gen failed for persona={persona['id']} subject={subject!r}: {exc}")
        return None
    text = str(resp or "").strip().strip('"')
    if not text:
        return None
    return {"persona": persona, "subject": subject, "comment": text}


def _upi_generate_pool(
    n_scenarios: int,
    num_personas: int,
    profiling_users: int,
    gen_model: str,
    seed: int,
    max_workers: int,
) -> tuple[dict[str, dict], list[dict]]:
    """Generate UPI entity scenarios AND persona profile groups.

    Returns (entity_scenarios, persona_groups) dicts matching the shipped
    user_entity.json and user_persona.json layouts.
    """
    rng = random.Random(seed)
    if n_scenarios > len(_UPI_CONTENT_SUBJECTS):
        print(f"UPI: requested {n_scenarios} scenarios but only "
              f"{len(_UPI_CONTENT_SUBJECTS)} subjects available; duplicating.")
        subjects = list(_UPI_CONTENT_SUBJECTS)
        while len(subjects) < n_scenarios:
            subjects += list(_UPI_CONTENT_SUBJECTS)
        subjects = subjects[:n_scenarios]
    else:
        subjects = rng.sample(_UPI_CONTENT_SUBJECTS, n_scenarios)

    # Build a flat job list: one LLM call per (scenario, persona)
    jobs: list[tuple[dict[str, Any], str, int]] = []  # (persona, subject, scenario_idx)
    scenario_personas: list[tuple[list[dict[str, Any]], dict[str, str]]] = []
    for s_idx, subject in enumerate(subjects):
        personas, primary = _upi_make_personas(num_personas, rng)
        scenario_personas.append((personas, primary))
        for p in personas:
            jobs.append((p, subject, s_idx))

    def _run(persona: dict[str, Any], subject: str, s_idx: int) -> dict[str, Any] | None:
        result = _upi_gen_one_comment(persona, subject, gen_model)
        if result is None:
            return None
        return {**result, "scenario_idx": s_idx}

    with llm.stage_scope("socialmaze_upi_gen"):
        results = llm.run_concurrent(
            _run, jobs, max_workers=max_workers,
            progress_desc="UPI comments",
        )

    # Reassemble per-scenario
    entity_scenarios: dict[str, dict[str, Any]] = {}
    # comments grouped by (age, gender) for persona groups
    from collections import defaultdict
    by_demo: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)

    per_scenario_comments: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for r in results:
        if r is None:
            continue
        persona = r["persona"]
        per_scenario_comments[r["scenario_idx"]].append({
            "persona": persona,
            "comment": r["comment"],
        })
        by_demo[(persona["age_group"], persona["gender"])].append({
            "comment": r["comment"],
            "subject": r["subject"],
        })

    for s_idx, subject in enumerate(subjects):
        personas, primary = scenario_personas[s_idx]
        entity_scenarios[f"scenario_{s_idx + 1}"] = {
            "product_name": subject,
            "primary_user_group": primary,
            "comments": per_scenario_comments.get(s_idx, []),
        }

    # Build persona groups: pick a demographic, sample `profiling_users` comments.
    valid_demos = {k: v for k, v in by_demo.items() if len(v) >= profiling_users}
    if not valid_demos:
        print("UPI: no demographic group has enough comments for a persona group; "
              "persona-flavour pool will be empty.")
        return entity_scenarios, []

    demo_keys = list(valid_demos.keys())
    persona_groups = []
    gen_rng = random.Random(seed + 1)
    for i in range(n_scenarios):
        key = gen_rng.choice(demo_keys)
        age, gender = key
        comments = valid_demos[key]
        sample = gen_rng.sample(comments, profiling_users) if len(comments) >= profiling_users else comments
        persona_groups.append({
            "group_id": i + 1,
            "demographics": {"age_group": age, "gender": gender},
            "comments": [
                {"product": c["subject"], "comment": c["comment"]} for c in sample
            ],
        })
    return entity_scenarios, persona_groups


def prepare_upi(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    entity_cache = out_dir / f"_pool_entity_n{args.pool_size}_seed{args.seed}.json"
    persona_cache = out_dir / f"_pool_persona_n{args.pool_size}_seed{args.seed}.json"

    if entity_cache.exists() and persona_cache.exists() and not args.regen:
        entity_scenarios = json.loads(entity_cache.read_text())
        persona_groups = json.loads(persona_cache.read_text())
        print(f"UPI: using cached pool "
              f"(entity={len(entity_scenarios)}, persona={len(persona_groups)}).")
    else:
        # Seed with shipped items (free); top up via generation.
        shipped_entity = load_upi_entity_shipped()
        shipped_persona = load_upi_persona_shipped()

        need = max(0, args.pool_size - max(len(shipped_entity), len(shipped_persona)))
        if need > 0:
            print(f"UPI: {len(shipped_entity)} shipped entity + "
                  f"{len(shipped_persona)} shipped persona; generating {need} more "
                  f"via {args.gen_model}...")
            gen_entity, gen_persona = _upi_generate_pool(
                n_scenarios=need,
                num_personas=args.num_personas,
                profiling_users=args.profiling_users,
                gen_model=args.gen_model,
                seed=args.seed,
                max_workers=args.max_workers,
            )
        else:
            gen_entity, gen_persona = {}, []

        # Merge: shipped keys first, then generated (renamed to avoid clashes)
        entity_scenarios = {f"shipped_{k}": v for k, v in shipped_entity.items()}
        for k, v in gen_entity.items():
            entity_scenarios[f"gen_{k}"] = v

        persona_groups = []
        for i, g in enumerate(shipped_persona):
            g = dict(g)
            g["group_id"] = f"shipped_{g.get('group_id', i)}"
            persona_groups.append(g)
        for g in gen_persona:
            g = dict(g)
            g["group_id"] = f"gen_{g.get('group_id')}"
            persona_groups.append(g)

        entity_cache.write_text(json.dumps(entity_scenarios, ensure_ascii=False, indent=2))
        persona_cache.write_text(json.dumps(persona_groups, ensure_ascii=False, indent=2))
        print(f"UPI: cached entity pool -> {entity_cache} "
              f"({len(entity_scenarios)} scenarios).")
        print(f"UPI: cached persona pool -> {persona_cache} "
              f"({len(persona_groups)} groups).")

    # Build instances per requested variant
    if args.variant == "persona":
        instances = [
            convert_upi_persona_item(g, idx=i) for i, g in enumerate(persona_groups)
        ]
    elif args.variant == "entity":
        instances = [
            convert_upi_entity_item(k, v) for k, v in entity_scenarios.items()
        ]
    else:  # both
        instances = [
            convert_upi_persona_item(g, idx=i) for i, g in enumerate(persona_groups)
        ] + [convert_upi_entity_item(k, v) for k, v in entity_scenarios.items()]

    print(f"UPI[{args.variant}]: {len(instances)} instances after conversion.")
    train, test = _split_pool(
        instances, train_n=args.train_n, test_n=args.test_n, seed=args.seed
    )
    _write_dataset(
        instances=train, task="upi", split_label="train", seed=args.seed,
        pool_total=len(instances),
        out_path=out_dir / f"train_n{len(train)}_seed{args.seed}.json",
        extra_meta={"variant": args.variant, "gen_model": args.gen_model},
    )
    _write_dataset(
        instances=test, task="upi", split_label="test", seed=args.seed,
        pool_total=len(instances),
        out_path=out_dir / f"test_n{len(test)}_seed{args.seed}.json",
        extra_meta={"variant": args.variant, "gen_model": args.gen_model},
    )


# CLI


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare SocialMaze datasets for SkillGen")
    subs = parser.add_subparsers(dest="task", required=True)

    # RDP
    pr = subs.add_parser("rdp", help="Review Decision Prediction (shipped 40-item debate.json)")
    pr.add_argument("--train-n", type=int, default=24)
    pr.add_argument("--test-n", type=int, default=16)
    pr.add_argument("--seed", type=int, default=42)
    pr.add_argument("--out-dir", default="data/socialmaze/rdp")
    pr.add_argument("--data-path", default=None,
                    help="Override path to debate.json (default: external submodule).")
    pr.set_defaults(func=prepare_rdp)

    # FTS
    pf = subs.add_parser("fts", help="Find the Spy (generates a pool via LLM)")
    pf.add_argument("--pool-size", type=int, default=120)
    pf.add_argument("--train-n", type=int, default=60)
    pf.add_argument("--test-n", type=int, default=50)
    pf.add_argument("--num-rounds", type=int, default=3)
    pf.add_argument("--gen-model", default="openai/gpt-5.4-nano")
    pf.add_argument("--max-workers", type=int, default=8)
    pf.add_argument("--seed", type=int, default=42)
    pf.add_argument("--out-dir", default="data/socialmaze/fts")
    pf.add_argument("--regen", action="store_true", help="Regenerate the pool even if cached.")
    pf.set_defaults(func=prepare_fts)

    # UPI
    pu = subs.add_parser("upi", help="User Profile Inference (generates a pool via LLM)")
    pu.add_argument("--pool-size", type=int, default=120)
    pu.add_argument("--train-n", type=int, default=60)
    pu.add_argument("--test-n", type=int, default=50)
    pu.add_argument("--variant", choices=("persona", "entity", "both"), default="persona")
    pu.add_argument("--num-personas", type=int, default=10,
                    help="Personas per scenario during generation.")
    pu.add_argument("--profiling-users", type=int, default=4,
                    help="Comments per persona group.")
    pu.add_argument("--gen-model", default="openai/gpt-5.4-nano")
    pu.add_argument("--max-workers", type=int, default=16)
    pu.add_argument("--seed", type=int, default=42)
    pu.add_argument("--out-dir", default="data/socialmaze/upi")
    pu.add_argument("--regen", action="store_true", help="Regenerate the pool even if cached.")
    pu.set_defaults(func=prepare_upi)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
