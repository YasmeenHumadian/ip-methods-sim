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
    "single": "Normal user (one person)",
    "shared": "Shared IP (office or campus, many people)",
    "attacker": "Obvious attacker",
    "low_and_slow": "Low-and-slow attacker (hides among normal users)",
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
st.caption("Mini simulation of two papers: Lewis et al. (2020) and BlackEye (Jeon & Tak, 2022)")

c1, c2, c3 = st.columns(3)
c1.markdown("**What we protect**  \nNormal users logging in, especially people behind a shared IP "
            "(a campus or an office). Block that IP and all of them are locked out.")
c2.markdown("**What we are defending against**  \nPassword-guessing attackers. Some are obvious, "
            "some are 'low and slow' and hide among normal users.")
c3.markdown("**The decision**  \nFor every IP: block or allow. The defender sees logs and "
            "blacklist flags, never the truth.")
st.markdown("**Two ways to decide.**  Lewis et al.: check blacklist databases and an address/location "
            "model.  BlackEye: learn from what the IP did in the logs, using cleaned-up labels from a human agent.")

tab1, tab2, tab3, tab4 = st.tabs(
    ["1. The world", "2. Who gets blocked", "3. Cleaning the labels", "4. The big simulation"]
)

# ---------- tab 1 ----------
with tab1:
    st.subheader("What is in this world")
    kind_counts = df["kind"].value_counts()
    cols = st.columns(4)
    for col, kind in zip(cols, KIND_NAMES):
        col.metric(KIND_NAMES[kind], f"{int(kind_counts.get(kind, 0)):,}")

    st.subheader("What the defender sees, and what is hidden")
    examples = pandas.concat([df[df["kind"] == k].head(1) for k in KIND_NAMES if (df["kind"] == k).any()])
    table = pandas.DataFrame({
        "Blacklists flagging it (of 4)": examples["db_hits"].to_numpy(),
        "Login attempts": examples["total_events"].to_numpy(),
        "Failed logins": (examples["failure_rate"] * 100).round(0).astype(int).astype(str).to_numpy() + "%",
        "Accounts tried": examples["distinct_accounts"].to_numpy(),
        "Night-time share": (examples["night_share"] * 100).round(0).astype(int).astype(str).to_numpy() + "%",
        "Truth (hidden from the methods)": examples["kind"].map(KIND_NAMES).to_numpy(),
    })
    st.dataframe(table, hide_index=True, use_container_width=True)
    st.caption("The methods see the first five columns. The last column is only used to score them.")

# ---------- tab 2 ----------
with tab2:
    st.subheader("Who gets blocked, and who gets wrongly blocked")
    chosen = st.multiselect(
        "Methods to compare",
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
        st.caption("Grey: baselines.  Blue: Lewis et al. (2020).  Green: BlackEye (Jeon & Tak, 2022).")

        show = r[["caught", "precision", "wrongly_blocked_share", "shared_wrongly_blocked_share"]].copy()
        show.columns = ["Attackers caught", "Right when it blocks (precision)",
                        "Innocent IPs wrongly blocked", "Innocent shared IPs wrongly blocked"]
        show.index = [m.replace("_", " ") for m in show.index]
        st.dataframe(show.style.format("{:.1%}"), use_container_width=True)

    full = results.set_index("method")
    a, b = full.loc["cross-check (Lewis)"], full.loc["logistic (BlackEye)"]
    st.info(
        f"In this world the Lewis cross-check catches {a['caught']:.0%} of attackers but wrongly blocks "
        f"{a['shared_wrongly_blocked_share']:.1%} of innocent shared IPs. BlackEye (logistic) catches "
        f"{b['caught']:.0%} and wrongly blocks {b['shared_wrongly_blocked_share']:.1%}."
    )
    st.caption("Rates, not counts, so methods are comparable: the ML methods are scored on the 30% of IPs "
               "they did not train on, the others on all IPs. One world only; the 10-seed averages are in run_log.md.")

# ---------- tab 3 ----------
with tab3:
    st.subheader("BlackEye trains on a human agent's labels, and cleans them first")
    st.write("The agent catches only about a quarter of attackers and sometimes blocks innocent IPs, so the "
             "training labels are wrong in both directions. BlackEye scores every IP with a first simple model, "
             "finds the point where the 'safe' and 'malicious' scores cross over, and drops labels on the wrong side.")
    k1, k2, k3 = st.columns(3)
    k1.metric("Labels that are wrong", counts["wrong_labels"], help=f"out of {counts['labels']:,} IPs")
    k2.metric("Labels dropped by cleansing", counts["dropped"])
    k3.metric("Dropped labels that really were wrong", counts["wrong_dropped"])

    ct = clean_table.set_index("cleansing")[["caught", "precision", "wrongly_blocked_share",
                                              "shared_wrongly_blocked_share"]]
    ct.index = ["Without cleansing", "With cleansing"]
    ct.columns = ["Attackers caught", "Right when it blocks (precision)",
                  "Innocent IPs wrongly blocked", "Innocent shared IPs wrongly blocked"]
    st.dataframe(ct.style.format("{:.1%}"), use_container_width=True)
    st.caption("BlackEye with logistic regression, scored on its 30% test IPs. Counts above are over all IPs.")

# ---------- tab 4 ----------
with tab4:
    st.subheader("The big simulation (planned, not built yet)")
    st.write("This mini version compares how the two papers decide whom to block. The big version adds time "
             "and asks the question neither paper answers: what happens after an innocent IP is blocked?")
    st.markdown(
        "- **Time:** logins arrive day by day, and IPs are blocked and listed over weeks.\n"
        "- **Time to blacklist:** BlackEye reports 66.2 days for the agent alone and 39.4 days with the model; the big version measures this.\n"
        "- **Recovery policies:** fixed expiry, decaying score, and probation, compared on how long innocent IPs stay blocked.\n"
        "- **Scale:** 30 seeds and about four experiments.\n"
        "- **Mini is a true subset:** the same world, methods and scoring, extended."
    )
