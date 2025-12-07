"""Replay the models on the competition's rounds, one CPU thread each (as on the server).

    uv run python tools/replay.py                                  # every model, every round
    uv run python tools/replay.py --models hybrid --rounds finals-2

Writes a row per run to results/replay.csv, each run's stakes to results/bets.csv and its
bankroll, whenever a settlement changed it, to results/bankroll.csv. A bet is keyed by the
game's row in the round's file (`game`); its odds and result stay in the organisers' data.

Models run with the settings they were last run with, except that the betting window is
opened up (`WHAT_IF`), so that every model bets in every round; for the finals submissions
that changes nothing in the round they were submitted to.
"""

import os

THREAD_VARS = "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"
for _var in THREAD_VARS:
    os.environ.setdefault(_var, "1")

import argparse
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

RESULTS = Path(__file__).resolve().parent.parent / "results"

# Settings changed from the models' presets so that they bet in every round.
WHAT_IF = {
    "hybrid": {"start_season": None},
    "simple-rolling": {"end_season": None},
}


def replay(model_name: str, round_name: str) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    import torch

    from qqh.environment import BET_COLS, ODDS_COLS, RESULT_COLS, Environment
    from qqh.models import MODELS
    from qqh.rounds import ROUNDS

    torch.set_num_threads(1)
    spec = ROUNDS[round_name]
    games = spec.games()
    model = MODELS[model_name].build(**WHAT_IF.get(model_name, {}))
    env = Environment(
        games,
        model,
        start_date=spec.start(games),
        init_bankroll=spec.bankroll,
        min_bet=spec.min_bet,
        max_bet=spec.max_bet,
    )
    started = time.process_time()
    result = env.run()
    seconds = time.process_time() - started

    settled = result.history[result.history["Cash_Invested"] == 0]["Bankroll"]
    changed = settled[settled.ne(settled.shift())]
    bankroll = changed.rename_axis("Date").reset_index()
    bets = result.bets[BET_COLS].rename_axis("game").reset_index()
    for frame in (bankroll, bets):
        frame.insert(0, "round", round_name)
        frame.insert(0, "model", model_name)

    stakes = result.bets[BET_COLS].values
    returns = (stakes * result.bets[RESULT_COLS].values * result.bets[ODDS_COLS].values).sum()
    row = {
        "model": model_name,
        "round": round_name,
        "final_bankroll": round(result.bankroll, 2),
        "profit": round(result.bankroll - spec.bankroll, 2),
        "bets": int((stakes > 0).sum()),
        "staked": round(float(stakes.sum()), 2),
        "returned": round(float(returns), 2),
        "lowest_bankroll": round(float(settled.min()), 2) if len(settled) else spec.bankroll,
        "cpu_seconds": round(seconds),
    }
    return row, bets, bankroll


def main() -> None:
    from qqh.models import MODELS
    from qqh.rounds import ROUNDS

    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=list(MODELS))
    parser.add_argument("--rounds", nargs="+", default=list(ROUNDS))
    parser.add_argument("--jobs", type=int, default=os.cpu_count())
    args = parser.parse_args()

    RESULTS.mkdir(exist_ok=True)
    paths = {name: RESULTS / f"{name}.csv" for name in ("replay", "bets", "bankroll")}
    tables = {
        name: pd.read_csv(path, float_precision="round_trip") if path.exists() else pd.DataFrame()
        for name, path in paths.items()
    }
    order = {name: i for i, name in enumerate(MODELS)}
    rounds = {name: i for i, name in enumerate(ROUNDS)}

    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        futures = [pool.submit(replay, m, r) for m in args.models for r in args.rounds]
        for future in as_completed(futures):
            row, bets, bankroll = future.result()
            print(
                f"{row['model']:>16} {row['round']:>13}  profit {row['profit']:>9.2f}"
                f"  bets {row['bets']:>4}  {row['cpu_seconds']:>5} s",
                flush=True,
            )
            new = {"replay": pd.DataFrame([row]), "bets": bets, "bankroll": bankroll}
            for name, table in tables.items():
                if not table.empty:
                    keep = (table["model"] != row["model"]) | (table["round"] != row["round"])
                    table = table[keep]
                if not new[name].empty:
                    table = pd.concat([table, new[name]], ignore_index=True)
                if table.empty:
                    continue
                # By model, then round; a run's rows keep their order.
                rank = table["model"].map(order) * len(rounds) + table["round"].map(rounds)
                tables[name] = table.iloc[rank.argsort(kind="stable")]
                tables[name].to_csv(paths[name], index=False)


if __name__ == "__main__":
    main()
