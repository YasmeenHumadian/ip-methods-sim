"""The six dots on the results slide: run them over several worlds and draw the chart.

No Streamlit in here, so it can be tested and reused outside the app.
"""
import pandas
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter

from simulator.evaluate import evaluate
from simulator.experiments import make_world, seeds_for
from simulator.methods_blackeye import blackeye, human_agent
from simulator.methods_lewis import crosscheck, lewis_ml

HUMAN = "Human agent"
CROSS = "Lewis cross-check"
LEWIS_ML = "Lewis ML"
JT_LOGISTIC = "Jeon & Tak logistic"
JT_FOREST = "Jeon & Tak random forest"
JT_NO_CLEAN = "Jeon & Tak logistic (no cleaning)"

# name -> (colour, filled, label offset in points, horizontal alignment)
STYLE = {
    HUMAN: ("#8b8b95", True, (10, 8), "left"),
    CROSS: ("#7b68b5", True, (-10, 8), "right"),
    LEWIS_ML: ("#7b68b5", True, (10, 8), "left"),
    JT_LOGISTIC: ("#1f9d6b", True, (-10, 12), "right"),
    JT_FOREST: ("#1f9d6b", True, (-10, -16), "right"),
    JT_NO_CLEAN: ("#1f9d6b", False, (-10, 10), "right"),
}
BEST, WORST = JT_LOGISTIC, LEWIS_ML


def headline_methods(df, cfg, seed):
    """Each method's answer on this world (True = blocked)."""
    return {
        HUMAN: human_agent(df, cfg, seed),
        CROSS: crosscheck(df, cfg),
        LEWIS_ML: lewis_ml(df, cfg, "logistic", seed=seed),
        JT_LOGISTIC: blackeye(df, cfg, "logistic", seed=seed),
        JT_FOREST: blackeye(df, cfg, "random_forest", seed=seed),
        JT_NO_CLEAN: blackeye(df, cfg, "logistic", use_cleansing=False, seed=seed),
    }


def run_headline(cfg):
    """Average of every method over the configured number of random worlds."""
    rows = []
    for seed in seeds_for(cfg):
        df = make_world(cfg, seed)
        for name, blocked in headline_methods(df, cfg, seed).items():
            row = evaluate(df, blocked)
            row["method"] = name
            rows.append(row)
    cols = ["caught", "shared_wrongly_blocked_share", "wrongly_blocked_share", "precision"]
    return pandas.DataFrame(rows).groupby("method", sort=False)[cols].mean()


def results_chart(res):
    """x = attackers caught (right is good), y = shared IPs wrongly blocked (low is good)."""
    fig = Figure(figsize=(8.2, 5.2))
    ax = fig.subplots()
    ymax = max(0.25, float(res["shared_wrongly_blocked_share"].max()) * 1.2)
    ax.set_xlim(0, 1.0)
    ax.set_ylim(-0.015, ymax)
    ax.axvspan(0.75, 1.0, ymin=0, ymax=0.2, color="#1f9d6b", alpha=0.08, lw=0)
    ax.text(0.995, 0.0, "best corner", ha="right", va="bottom", fontsize=9, color="#1f9d6b", alpha=0.8,
            transform=ax.get_xaxis_transform())

    for name, (colour, filled, offset, ha) in STYLE.items():
        x, y = res.loc[name, "caught"], res.loc[name, "shared_wrongly_blocked_share"]
        ax.scatter([x], [y], s=190, facecolor=colour if filled else "white", edgecolor=colour,
                   linewidth=2.2, zorder=3)
        ax.annotate(name, (x, y), xytext=offset, textcoords="offset points", ha=ha, fontsize=9.5, color="#1b1a2e")

    ax.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.set_xlabel("Attackers caught  (further right is better)", fontsize=11)
    ax.set_ylabel("Shared IPs wrongly blocked  (lower is better)", fontsize=11)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(alpha=0.25)

    ax.legend(
        handles=[
            Line2D([], [], marker="o", ls="", color="#8b8b95", markersize=9, label="Human agent"),
            Line2D([], [], marker="o", ls="", color="#7b68b5", markersize=9, label="Lewis et al."),
            Line2D([], [], marker="o", ls="", color="#1f9d6b", markersize=9, label="Jeon & Tak (cleaned labels)"),
            Line2D([], [], marker="o", ls="", markerfacecolor="white", markeredgecolor="#1f9d6b",
                   markeredgewidth=2, markersize=9, label="Jeon & Tak, cleaning off"),
        ],
        loc="upper left", frameon=False, fontsize=9.5,
    )
    fig.tight_layout()
    return fig
