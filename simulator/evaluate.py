"""One way to score every method, so the comparison is fair."""
import numpy as np
import pandas


def _rows(df, blocked):
    """ML methods return only their test IPs (a Series); the others cover every IP."""
    if isinstance(blocked, pandas.Series):
        return df.loc[blocked.index]
    return df


def evaluate(df, blocked):
    rows = _rows(df, blocked)
    flags = np.asarray(blocked, dtype=bool)
    bad = rows["is_malicious"].to_numpy()
    shared = (rows["kind"] == "shared").to_numpy()

    tp = int((flags & bad).sum())
    fp = int((flags & ~bad).sum())
    fn = int((~flags & bad).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    caught = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * caught / (precision + caught) if precision + caught else 0.0
    innocent_shared = shared & ~bad

    return {
        "caught": caught,                                   # share of attackers blocked (recall)
        "precision": precision,                             # share of blocks that were right
        "f1": f1,
        "wrongly_blocked": fp,                              # innocent IPs blocked
        "wrongly_blocked_share": flags[~bad].mean(),        # share of innocent IPs blocked
        "shared_wrongly_blocked_share": flags[innocent_shared].mean() if innocent_shared.any() else 0.0,
        "n_blocked": int(flags.sum()),
    }


def agent_wrong_blocks_corrected(df, agent, blocked):
    """Of the innocent IPs the human agent wrongly blocked, the share this method does NOT block."""
    rows = _rows(df, blocked)
    agent_flags = pandas.Series(np.asarray(agent, dtype=bool), index=df.index).loc[rows.index].to_numpy()
    wrong = agent_flags & ~rows["is_malicious"].to_numpy()
    if wrong.sum() == 0:
        return float("nan")
    return float((~np.asarray(blocked, dtype=bool)[wrong]).mean())