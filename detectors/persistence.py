"""Persistence rule for the statistical trigger (Revision v3, option b).

Rule: alarm only if the statistical trigger fires in `k` of the last
`n` windows (the current window included). The rule is a pure
post-processing of the per-window trigger stream; it does not change
any statistic, threshold or test.

Semantics (deliberately simple and streaming-realistic):

* WINDOWS: the trailing sequence of detector decisions. For lag
  streams this is the stream's own evaluated windows. For isolated
  matrix windows the harness reconstructs the same continuous stream:
  the window itself plus the preceding `n - 1` raw (clean) blocks,
  because the Phase 7 harness never writes injections back to the raw
  dataset.
* START-UP: with fewer than `n` windows of history the alarm requires
  all `k` triggers to be inside the available windows; a single
  isolated trigger cannot alarm when k >= 2.
* COST: a sustained drift alarms on the `k`-th consecutive trigger
  window, so the honest added detection lag is `(k - 1)` windows.

The k/n values come from configs/thresholds.yaml
(`statistical.persistence.{enabled,k,n}`), selected on validation
blocks only (experiments/revision3_fpr_options.py).
"""

SECTION_KEY = "persistence"


def persistence_last_n(triggers, n):
    """Return the last `n` trigger booleans, oldest first."""
    return list(triggers)[-int(n):] if n > 0 else []


def persistence_alarm(triggers, k, n):
    """True iff the trigger fired in at least `k` of the last `n` windows.

    `triggers` is a sequence of per-window trigger booleans, oldest
    first; the LAST element must be the CURRENT window.
    """
    window = persistence_last_n(triggers, n)
    if len(window) < k:
        return False
    return sum(1 for fired in window if fired) >= int(k)


def added_lag_windows(k):
    """Honest added detection lag: a sustained drift alarms on the
    k-th trigger window, i.e. `(k - 1)` windows after the first."""
    return max(0, int(k) - 1)
