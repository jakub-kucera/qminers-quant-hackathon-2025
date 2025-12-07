"""Chart results/replay.csv: profit per model in each finals round, the submission marked.

    uv run --group analysis python tools/plot_results.py

Writes results/replay_light.png and results/replay_dark.png (the README shows the one that
matches the reader's GitHub theme).
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.ticker import FuncFormatter, MultipleLocator  # noqa: E402

from qqh.models import MODELS  # noqa: E402
from qqh.rounds import ON_THE_SERVER, ROUNDS, SUBMITTED  # noqa: E402

RESULTS = Path(__file__).resolve().parent.parent / "results"
FINALS = ["finals-1", "finals-2", "finals-3"]

# One validated series blue per theme (dataviz reference palette, slot 1) and its chrome.
THEMES = {
    "light": dict(
        surface="#fcfcfb", series="#2a78d6", ink="#0b0b0b", secondary="#52514e",
        muted="#898781", grid="#e1e0d9", baseline="#c3c2b7",
    ),
    "dark": dict(
        surface="#1a1a19", series="#3987e5", ink="#f0efec", secondary="#c3c2b7",
        muted="#898781", grid="#2c2c2a", baseline="#383835",
    ),
}  # fmt: skip


def draw(table: pd.DataFrame, theme: dict) -> plt.Figure:
    models = list(MODELS)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.4), sharey=True, facecolor=theme["surface"])
    lo = min(table.loc[table["round"].isin(FINALS), "profit"].min(), 0)
    hi = max(table.loc[table["round"].isin(FINALS), "profit"].max(), 0)
    pad = (hi - lo) * 0.22
    for ax, round_name in zip(axes, FINALS, strict=True):
        ax.set_facecolor(theme["surface"])
        rows = table[table["round"] == round_name].set_index("model").reindex(models)
        y = range(len(models))
        ax.barh(y, rows["profit"], height=0.32, color=theme["series"], zorder=2)
        ax.axvline(0, color=theme["baseline"], linewidth=1, zorder=1)
        ax.grid(axis="x", color=theme["grid"], linewidth=0.6, zorder=0)
        ax.set_xlim(lo - pad, hi + pad)
        for i, (name, profit) in enumerate(rows["profit"].items()):
            label = f"{profit:+,.0f}"
            ax.text(
                profit + (pad * 0.06 if profit >= 0 else -pad * 0.06),
                i,
                label,
                va="center",
                ha="left" if profit >= 0 else "right",
                fontsize=8.5,
                color=theme["ink"] if name == SUBMITTED[round_name] else theme["secondary"],
                fontweight="bold" if name == SUBMITTED[round_name] else "normal",
            )
        seasons = ROUNDS[round_name].seasons
        ax.set_title(
            f"Finals round {round_name[-1]}, seasons {seasons[0]}-{seasons[1]}",
            fontsize=10,
            color=theme["ink"],
            loc="left",
            pad=22,
        )
        ax.text(
            0,
            1.03,
            f"submitted: {SUBMITTED[round_name]} (on the server: {ON_THE_SERVER[round_name]})",
            transform=ax.transAxes,
            fontsize=8.5,
            color=theme["secondary"],
        )
        ax.xaxis.set_major_locator(MultipleLocator(1000))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:+,.0f}" if v else "0"))
        ax.tick_params(colors=theme["muted"], labelsize=8.5, length=0)
        for side in ("top", "right", "bottom", "left"):
            ax.spines[side].set_visible(False)
        ax.set_xlabel("profit, replayed on the published data", fontsize=8.5, color=theme["muted"])
    axes[0].set_yticks(range(len(models)), models, color=theme["secondary"])
    axes[0].invert_yaxis()
    fig.tight_layout()
    return fig


def main() -> None:
    table = pd.read_csv(RESULTS / "replay.csv")
    for name, theme in THEMES.items():
        fig = draw(table, theme)
        fig.savefig(RESULTS / f"replay_{name}.png", dpi=160, facecolor=theme["surface"])
        plt.close(fig)


if __name__ == "__main__":
    main()
