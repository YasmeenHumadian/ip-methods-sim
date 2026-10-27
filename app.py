"""Simple demo: press Run, see how each method did.

Mini simulation of Lewis et al. (2020) and Jeon & Tak (2022).
Run with:  streamlit run app.py        (the full four-tab version is app_full.py)
"""
import streamlit as st

from simulator.config import load_config
from simulator.demo import BEST, WORST, results_chart, run_headline

st.set_page_config(page_title="Wrongly blocked IPs", layout="wide")


@st.cache_data(show_spinner=False)
def simulate(n_ips, attacker_pct, shared_pct):
    cfg = load_config()
    cfg["world"]["n_ips"] = n_ips
    cfg["world"]["malicious_share"] = attacker_pct / 100
    cfg["world"]["shared_share"] = shared_pct / 100
    return run_headline(cfg)


st.title("Wrongly blocked IP addresses")
st.caption("Mini simulation of two papers: Lewis et al. (2020) and Jeon & Tak (2022)")

with st.expander("The fake world (optional: change it)"):
    n_ips = st.slider("Number of IP addresses", 500, 5000, 2000, step=500)
    attacker_pct = st.slider("Attackers (% of IPs)", 5, 20, 10)
    shared_pct = st.slider("Shared IPs (% of innocent IPs)", 10, 60, 40)

if st.button("Run simulation", type="primary"):
    with st.spinner("Running 10 random worlds..."):
        st.session_state["res"] = simulate(n_ips, attacker_pct, shared_pct)
        st.session_state["world"] = (n_ips, attacker_pct, shared_pct)

res = st.session_state.get("res")
if res is None:
    st.info("Press **Run simulation** to build a fake world and test every method on it.")
else:
    n, a, s = st.session_state["world"]
    st.caption(f"Fake world: {n:,} IPs, {a}% attackers, {s}% of innocent IPs shared. "
               "Averaged over 10 random worlds.")

    left, right = st.columns([3, 1])
    with left:
        st.pyplot(results_chart(res))
    with right:
        for title, name in (("Best", BEST), ("Worst", WORST)):
            st.markdown(f"**{title}: {name}**")
            st.metric("Attackers caught", f"{int(round(res.loc[name, 'caught'] * 1000) / 10 + 0.5)}%")  # round half up, as on the slide
            st.metric("Shared IPs wrongly blocked", f"{res.loc[name, 'shared_wrongly_blocked_share']:.1%}")

    with st.expander("All the numbers"):
        table = res[["caught", "shared_wrongly_blocked_share"]].copy()
        table.columns = ["Attackers caught", "Shared IPs wrongly blocked"]
        st.dataframe(table.style.format("{:.1%}"), use_container_width=True)
        st.caption("Fake data from our own assumed world, so the results show how the methods compare here, "
                   "not how they would perform on real traffic.")
