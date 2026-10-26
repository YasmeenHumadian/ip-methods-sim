"""Streamlit demo: wrongly blocked IPs, Lewis et al. (2020) vs BlackEye (Jeon & Tak, 2022).
Run with:  streamlit run app.py
"""
import matplotlib.pyplot as plt
import pandas
import streamlit as st
from matplotlib.ticker import PercentFormatter

from simulator.config import load_config
from simulator.experiments import cleansing_demo, run_once

KIND_NAMES = {
    "single": "Normal user",
    "shared": "Shared IP",
    "attacker": "Obvious attacker",
    "low_and_slow": "Low-and-slow attacker",
}
DEFAULT_METHODS = [
    "human agent",
    "cross-check (Lewis)",
    "logistic (Lewis ML)",
    "logistic (BlackEye)",
]

st.set_page_config(page_title="Wrongly blocked IPs", layout="wide")


@st.cache_data(show_spinner="Simulating...")
def simulate(n_ips, attacker_pct, shared_pct, seed):
    cfg = load_config()
    cfg["world"]["n_ips"] = n_ips
    cfg["world"]["malicious_share"] = attacker_pct / 100
    cfg["world"]["shared_share"] = shared_pct / 100
    cfg["seed"] = seed
    df, results = run_once(cfg, seed)
    counts, clean_table = cleansing_demo(cfg, seed)
    return df, results, counts, clean_table


def colour(name):
    if "BlackEye" in name:
        return "#1a9850"
    if "Lewis" in name:
        return "#3b6fb6"
    return "#9aa0a6"


def bars(ax, labels, values, colours, title, fmt, full_scale=False):
    ax.barh(labels, values, color=colours)
    ax.set_title(title, loc="left", fontsize=12)
    top = max(values) if max(values) > 0 else 1.0
    if full_scale:  # a share of attackers: always 0% to 100%
        ax.set_xlim(0, 1.12)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    else:
        ax.set_xlim(0, max(0.05, top * 1.3))
    ax.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    for i, v in enumerate(values):
        ax.text(v + top * 0.02, i, format(v, fmt), va="center", fontsize=10)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


# ---------- sidebar ----------
with st.sidebar.form("controls"):
    st.header("The world")
    n_ips = st.slider("Number of IP addresses", 500, 5000, 2000, step=500)
    attacker_pct = st.slider("Attackers (% of IPs)", 5, 20, 10)
    shared_pct = st.slider("Shared IPs (% of innocent IPs)", 10, 60, 40)
    seed = st.number_input("Random seed", 1, 999, 1)
    st.form_submit_button("Run simulation", type="primary")

df, results, counts, clean_table = simulate(n_ips, attacker_pct, shared_pct, int(seed))

# ---------- header ----------
st.title("Wrongly blocked IP addresses")
st.caption("Mini simulation: Lewis et al. (2020) vs Jeon & Tak (2022)")

c1, c2, c3 = st.columns(3)
c1.markdown("**Protect**  \nNormal users, including people sharing one IP")
c2.markdown("**Defend against**  \nPassword-guessing attackers")
c3.markdown("**Decide**  \nBlock or allow each IP, without seeing the truth")
st.markdown("**Lewis et al. (2020):** blacklists and address  ·  **Jeon & Tak (2022):** login behaviour")

tab1, tab2, tab3 = st.tabs(["1. The world", "2. Who gets blocked", "3. Cleaning the labels"])

# ---------- tab 1 ----------
with tab1:
    st.subheader("The world")
    kind_counts = df["kind"].value_counts()
    cols = st.columns(4)
    for col, kind in zip(cols, KIND_NAMES):
        col.metric(KIND_NAMES[kind], f"{int(kind_counts.get(kind, 0)):,}")

    st.subheader("What the defender sees")
    examples = pandas.concat([df[df["kind"] == k].head(1) for k in KIND_NAMES if (df["kind"] == k).any()])
    table = pandas.DataFrame({
        "Blacklists (of 4)": examples["db_hits"].to_numpy(),
        "Login attempts": examples["total_events"].to_numpy(),
        "Failed logins": (examples["failure_rate"] * 100).round(0).astype(int).astype(str).to_numpy() + "%",
        "Accounts tried": examples["distinct_accounts"].to_numpy(),
        "Night-time share": (examples["night_share"] * 100).round(0).astype(int).astype(str).to_numpy() + "%",
        "Truth (hidden)": examples["kind"].map(KIND_NAMES).to_numpy(),
    })
    st.dataframe(table, hide_index=True, use_container_width=True)

# ---------- tab 2 ----------
with tab2:
    st.subheader("Who gets blocked")
    chosen = st.multiselect(
        "Methods",
        list(results["method"]),
        default=DEFAULT_METHODS,
        format_func=lambda s: s.replace("_", " "),
    )
    if not chosen:
        st.warning("Pick at least one method.")
    else:
        r = results.set_index("method").loc[chosen]
        labels = [m.replace("_", " ") for m in r.index][::-1]
        colours = [colour(m) for m in r.index][::-1]
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 0.6 * len(chosen) + 1.6), sharey=True)
        bars(a1, labels, list(r["caught"])[::-1], colours, "Attackers caught (higher is better)", ".0%",
             full_scale=True)
        bars(a2, labels, list(r["shared_wrongly_blocked_share"])[::-1], colours,
             "Innocent shared IPs wrongly blocked (lower is better)", ".1%")
        fig.tight_layout()
        st.pyplot(fig)
        st.caption("Grey: baseline  ·  Blue: Lewis et al.  ·  Green: Jeon & Tak")

        show = r[["caught", "precision", "wrongly_blocked_share", "shared_wrongly_blocked_share"]].copy()
        show.columns = ["Attackers caught", "Precision", "Innocent wrongly blocked", "Shared wrongly blocked"]
        show.index = [m.replace("_", " ") for m in show.index]
        st.dataframe(show.style.format("{:.1%}"), use_container_width=True)

    full = results.set_index("method")
    a, b = full.loc["cross-check (Lewis)"], full.loc["logistic (BlackEye)"]
    st.info(
        f"Cross-check: {a['caught']:.0%} caught, {a['shared_wrongly_blocked_share']:.1%} of innocent shared IPs wrongly blocked.  "
        f"BlackEye: {b['caught']:.0%} caught, {b['shared_wrongly_blocked_share']:.1%} wrongly blocked."
    )
    st.caption("One world. 10-seed averages are in run_log.md.")

# ---------- tab 3 ----------
with tab3:
    st.subheader("Cleaning the labels")
    st.write("The human agent's labels are wrong in both directions. BlackEye drops the ones that look wrong.")
    k1, k2, k3 = st.columns(3)
    k1.metric("Wrong labels", counts["wrong_labels"], help=f"out of {counts['labels']:,} IPs")
    k2.metric("Dropped by cleansing", counts["dropped"])
    k3.metric("Dropped and really wrong", counts["wrong_dropped"])

    ct = clean_table.set_index("cleansing")[["caught", "precision", "wrongly_blocked_share",
                                              "shared_wrongly_blocked_share"]]
    ct.index = ["Without cleansing", "With cleansing"]
    ct.columns = ["Attackers caught", "Precision", "Innocent wrongly blocked", "Shared wrongly blocked"]
    st.dataframe(ct.style.format("{:.1%}"), use_container_width=True)
    st.caption("BlackEye logistic, on test IPs")
