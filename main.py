"""gta6-trophy-rarity-forecast-01: earners-over-players power-law forecaster.

Forecasts, for each of GTA VI's 55 PS5 trophies, five quantiles of its earn
rate on Exophase at the case's target, as a program that reads one case file
(/input/<id>.in) and writes /output/<id>.out. Python 3.13, standard library
only, no network, deterministic.

Model
-----
An earn rate is a ratio: rate = 100 * earners / players_tracked. Both counts
only move when Exophase refreshes the page, so a case's history is a step
function: long flat stretches, then a jump. The forecaster works on the
counts, not the rates:

  * it collapses the history into distinct states (consecutive identical
    reads are one state, kept at the time it first appeared);
  * it fits, over the last 24 hours of distinct states, a least-squares line
    of ln(players_tracked) against ln(1 + hours since launch), and one line
    per trophy of ln(earners) against the same clock (earners recovered as
    rate * players_tracked / 100);
  * each trophy's slope is shrunk toward the cross-trophy median slope by
    empirical-Bayes weights: a slope the data pin down precisely (small
    standard error) is trusted almost fully, a noisy one is pulled hard
    toward the median, so a trophy with two noisy points cannot extrapolate
    wildly while a clean fast-growing trophy keeps its own slope. The
    standard error is inflated by the page's rounding (rates print to 2
    decimals, so a trophy at 0.01% carries ~50% relative noise in its
    recovered earners and its fitted slope would otherwise just mirror the
    tracked-players slope);
  * it reads both lines at the target, divides, and clamps the rate to
    [0.01, 99.99]. Earners never decrease; a trophy's forecast earners never
    exceed the forecast players tracked.

Two cases the trend fit cannot see need priors, both stated in the
competition's own rules:

  * A history with a single distinct state (launch night: the page sat
    unchanged at 638 players tracked for hours) has no trend to fit, yet a
    refresh is coming: the rules record the first refresh taking tracked
    players 638 -> 3,868. The forecaster therefore grows tracked players by
    a power law in log time since launch, N(T) = N_last * (1 + h_T) ** 1.05,
    which reproduces that first refresh (638 * 5.5 ** 1.05 = 3,820, against
    the recorded 3,868), and grows each trophy's earners by a small
    category prior (early-story trophies faster than late/grind ones), so the
    forecast rate is the diluted rate, not the stale one.
  * A trophy already at ~100% (the opening mission, earned by essentially
    every tracked player) stays there: its forecast rate is its last rate
    with a narrow spread. A trophy with no earners anywhere stays at the
    0.01% floor with a wide spread.

Spreads
-------
Each trophy's logit spread combines the residual scatter of its earners fit
and of the players fit (both inflated by how far the target sits beyond the
fitted clock range), a base term that grows with the horizon in days, and a
term for a stale last state. The spread is floored so the 90% band stays
honest and capped so one wild trophy cannot blow out the mean loss. The
quantiles are the normal quantiles of the forecast logit at the five scored
levels, mapped back to percent and rounded to 4 decimals.

Validation (leave-the-future-out, scored exactly as the contest scores: mean
pinball loss on the logit scale; lower is better):
  * backtest B: history = the practice case p01's flat launch-night reads
    (checkpoint 2026-11-19T03:30Z), target = the live Exophase page
    2026-11-19T22:17Z (18.8 h later, several refreshes, tracked players
    638 -> 10,000+): this model 0.20, house persistence baseline 2.31,
    house curve baseline 2.31 (both flat on a flat history).
  * synthetic final-like case: a simulated multi-refresh history over
    59.5 h (tracked players 638 -> 45,982, 17 distinct states), target
    4.5 days out: this model 0.10, house persistence baseline 0.36.
    (The synthetic world uses the same power-law family as the model, so
    this number is optimistic; the real world will deviate, which is what
    the spreads are for.)
  * practice cases p00/p01: the program writes valid forecasts; their
    server score is the practice loss (the practice pays nothing).
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first, so `import model`
# finds a model.py shipped beside this file.
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

# The quantile levels the contest scores, in the order each row lists them.
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
# The standard normal's quantile at each level.
Z = [-1.645, -0.674, 0, 0.674, 1.645]

# Launch reference: the house curve baseline measures hours from here.
LAUNCH = datetime(2026, 11, 19, 0, 0, 0, tzinfo=timezone.utc)

# --- model constants (see the module docstring for why each is what it is) ---
# Power-law exponent for tracked-players growth when the history is flat:
# reproduces the first recorded refresh (638 -> 3,868 by 04:30Z).
FLAT_BETA_N = 1.05
# Earner-growth exponent by trophy category, flat-history branch only.
GAMMA_FAST = 0.8    # opening story beats: chapter 1, first missions, quick side activities
GAMMA_MID = 0.3     # later chapters and mid-game missions
GAMMA_SLOW = 0.1    # endgame / grind: final mission, 100%, 250 stunts, platinum
GAMMA_DEFAULT = 0.2
# Fit window: only the last 24 h of distinct states drive the trend fit.
WINDOW_HOURS = 24.0
# Empirical-Bayes shrinkage: the cross-trophy spread of true slopes. A
# trophy's own slope is weighted tau^2 / (tau^2 + se^2) against the median.
TAU_SLOPE = 0.25
# Slope clamps (log earners per log hour).
SLOPE_MIN = -0.5
SLOPE_MAX = 3.0
# Spread pieces, in logit units.
SPREAD_FLOOR = 0.35
SPREAD_CAP = 2.0
SPREAD_FLAT = 0.65      # flat history: the refresh size is genuinely uncertain
SPREAD_NEAR100 = 0.15   # a ~100% trophy barely moves
SPREAD_ZERO = 0.8       # a trophy with no earners yet
HORIZON_SIGMA = 0.20    # base spread per sqrt(day) of horizon
STALE_SIGMA = 0.15      # extra when the last distinct state is over 2 h old
RESID_FLOOR = 0.15
RESID_CAP = 1.0


def parse_time(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def clamp(x: float) -> float:
    return min(99.99, max(0.01, x))


def logit(pct: float) -> float:
    p = clamp(pct)
    return math.log(p / (100 - p))


def inv_logit(lg: float) -> float:
    if lg >= 0:
        return 100 / (1 + math.exp(-lg))
    e = math.exp(lg)
    return 100 * e / (1 + e)


def round4(x: float):
    """x rounded to 4 decimals, half up, as JavaScript's Math.round does it."""
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if value == int(value) else value


def ols(xs, ys):
    """Least squares y = a + b*x.

    Returns (a, b, resid_std, se_slope, sxx, mean_x, mean_y): the means let a
    shrunk slope be re-centred on the data instead of pivoting at x = 0.
    """
    n = len(xs)
    if n == 0:
        return None
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 1e-12:
        return (my, 0.0, 0.0 if n > 1 else None, None, sxx, mx, my)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    b = sxy / sxx
    a = my - b * mx
    if n > 2:
        resid = math.sqrt(sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys)) / (n - 2))
    else:
        resid = None
    se = (resid / math.sqrt(sxx)) if resid is not None else None
    return (a, b, resid, se, sxx, mx, my)


def median(values):
    v = sorted(values)
    n = len(v)
    if n == 0:
        return 0.0
    return v[n // 2] if n % 2 else 0.5 * (v[n // 2 - 1] + v[n // 2])


def gamma_for(trophy):
    """Earner-growth exponent prior, flat-history branch only."""
    text = (trophy.get("name", "") + " " + trophy.get("description", "")).lower()
    if "gone fishing" in text or "chapter 1" in text or "breakfast run" in text:
        return GAMMA_FAST
    if "fishing challenge" in text or "arcade game" in text or "discover 100 vehicles" in text:
        return GAMMA_FAST
    if "chapter 2" in text:
        return 0.5
    if "chapter 3" in text or "chapter 4" in text:
        return GAMMA_MID
    if "final mission" in text or "100% completion" in text or "250 stunts" in text:
        return GAMMA_SLOW
    if trophy.get("tier") == "platinum":
        return GAMMA_SLOW
    return GAMMA_DEFAULT


def distinct_states(case):
    """Collapse consecutive identical reads into states.

    Returns a list of (at_datetime, players_tracked, rates) in time order,
    one per distinct page state, kept at the time it first appeared.
    """
    states = []
    last_key = None
    for read in case.get("history") or []:
        rates = read.get("rates") or []
        key = (read.get("players_tracked"), tuple(rates))
        if key == last_key:
            continue
        last_key = key
        states.append((parse_time(read["at"]), read.get("players_tracked"), rates))
    return states


def forecast_case(case: dict, name: str) -> dict:
    if case["levels"] != LEVELS:
        raise ValueError(f"{name} asks for levels {case['levels']}, not {LEVELS}")
    trophies = case["trophies"]
    n_tr = len(trophies)
    horizon_days = case["horizon_minutes"] / 1440.0
    at = parse_time(case["at"])
    target = parse_time(case["target"])
    h_target = (target - LAUNCH).total_seconds() / 3600.0
    x_target = math.log1p(max(h_target, 0.0))

    states = distinct_states(case)
    if not states:
        # No history at all: a neutral, very wide forecast.
        return {"trophies": [
            {"id": t["id"], "q": [round4(v) for v in (0.05, 0.2, 0.5, 1.3, 3.0)]}
            for t in trophies
        ]}

    last_at, last_n, last_rates = states[-1]
    last_n = last_n or 0
    # Earners in the last state, per trophy.
    last_e = []
    for i in range(n_tr):
        r = last_rates[i] if i < len(last_rates) else None
        last_e.append(None if r is None or last_n == 0 else r * last_n / 100.0)

    h_last = (last_at - LAUNCH).total_seconds() / 3600.0
    x_last = math.log1p(max(h_last, 0.0))
    stale_hours = (at - last_at).total_seconds() / 3600.0

    flat = len(states) == 1
    if flat:
        # --- flat-history branch: the refresh prior (see module docstring) ---
        h_stale = max((states[0][0] - LAUNCH).total_seconds() / 3600.0, 0.0)
        n_target = last_n * ((1.0 + max(h_target, 0.0)) / (1.0 + h_stale)) ** FLAT_BETA_N
        n_target = max(n_target, float(last_n))
        rows = []
        for i, trophy in enumerate(trophies):
            r_last = last_rates[i] if i < len(last_rates) else None
            if r_last is not None and r_last >= 90.0:
                med = r_last
                spread = SPREAD_NEAR100
            elif last_e[i] is None or last_e[i] <= 0:
                med = 0.01
                spread = SPREAD_ZERO
            else:
                g = gamma_for(trophy)
                e_target = last_e[i] * ((1.0 + max(h_target, 0.0)) / (1.0 + max(h_last, 0.0))) ** g
                e_target = min(e_target, 0.9999 * n_target)
                med = clamp(100.0 * e_target / n_target)
                spread = SPREAD_FLAT
            lg = logit(med)
            q = [round4(clamp(inv_logit(lg + z * spread))) for z in Z]
            q = sorted(q)
            rows.append({"id": trophy["id"], "q": q})
        return {"trophies": rows}

    # --- trend branch: power-law fits on the last WINDOW_HOURS of states ---
    window_start = at.timestamp() - WINDOW_HOURS * 3600.0
    window = [s for s in states if s[0].timestamp() >= window_start]
    if len(window) < 3:
        window = states
    xs = [math.log1p(max((s[0] - LAUNCH).total_seconds() / 3600.0, 0.0)) for s in window]
    x_range = (max(xs) - min(xs)) if len(xs) > 1 else 1.0
    if x_range <= 1e-9:
        x_range = 1.0

    # Tracked-players fit.
    ln_n = [math.log(s[1]) for s in window if s[1] and s[1] > 0]
    xs_n = [x for s, x in zip(window, xs) if s[1] and s[1] > 0]
    fit_n = ols(xs_n, ln_n) if len(xs_n) >= 2 else None
    if fit_n is None:
        a_n, b_n, s_n = math.log(max(last_n, 1)), 0.0, 0.5
        mx_n, my_n = xs_n[-1] if xs_n else 0.0, math.log(max(last_n, 1))
    else:
        a_n, b_n, s_n, _se_n, _sxx_n, mx_n, my_n = fit_n
        b_n = min(max(b_n, 0.0), 2.5)
        s_n = 0.5 if s_n is None else min(max(s_n, RESID_FLOOR), RESID_CAP)
    ln_n_target = my_n + b_n * (x_target - mx_n)
    n_target = max(math.exp(ln_n_target), float(last_n))

    # Per-trophy earner fits, shrunk toward the cross-trophy median slope.
    slopes = []
    fits = []
    for i in range(n_tr):
        pts = []
        for s, x in zip(window, xs):
            rates = s[2]
            if i < len(rates) and rates[i] is not None and s[1]:
                e = rates[i] * s[1] / 100.0
                if e > 0:
                    pts.append((x, math.log(e), rates[i]))
        if len(pts) >= 2 and len({p[0] for p in pts}) >= 2:
            a_i, b_i, s_i, se_i, sxx_i, mx_i, my_i = ols([p[0] for p in pts], [p[1] for p in pts])
            fits.append((a_i, b_i, s_i, se_i, sxx_i, mx_i, my_i, len(pts), median([p[2] for p in pts])))
            slopes.append(b_i)
        else:
            fits.append(None)
    b_med = median(slopes) if slopes else 0.0
    b_med = min(max(b_med, SLOPE_MIN), SLOPE_MAX)
    # Cross-trophy spread of the fitted slopes, for the EB weights.
    if len(slopes) >= 3:
        m_sl = sum(slopes) / len(slopes)
        tau = math.sqrt(sum((b - m_sl) ** 2 for b in slopes) / (len(slopes) - 1))
    else:
        tau = TAU_SLOPE
    tau = max(tau, 0.15)

    # Extrapolation inflation: how far the target sits beyond the fitted range.
    dx = max(x_target - max(xs), 0.0)
    infl = 1.0 + (dx / x_range) ** 2
    horizon_var = (HORIZON_SIGMA ** 2) * horizon_days
    stale_var = STALE_SIGMA ** 2 if stale_hours > 2.0 else 0.0

    rows = []
    for i, trophy in enumerate(trophies):
        r_last = last_rates[i] if i < len(last_rates) else None
        if r_last is not None and r_last >= 90.0:
            med = r_last
            spread = SPREAD_NEAR100
        elif last_e[i] is None or last_e[i] <= 0:
            med = 0.01
            spread = SPREAD_ZERO
        else:
            fit = fits[i]
            if fit is None:
                # No usable earner history: earners flat, median = diluted last rate.
                e_target = last_e[i]
                var = (0.5 ** 2) * infl + horizon_var + stale_var
            else:
                a_i, b_i, s_i, se_i, sxx_i, mx_i, my_i, n_i, r_med = fit
                b_i = min(max(b_i, SLOPE_MIN), SLOPE_MAX)
                # Empirical-Bayes weight on the trophy's own slope. The
                # slope's standard error is inflated by the page's rounding:
                # rates print to 2 decimals, so a trophy at 0.01% carries
                # ~50% relative noise in its recovered earners and its fitted
                # slope is mostly the tracked-players slope in disguise.
                se2 = (se_i ** 2) if se_i is not None else 0.09
                q_noise = 0.005 / max(r_med, 0.01)
                if sxx_i and sxx_i > 0:
                    se2 += (q_noise ** 2) / sxx_i
                w = (tau ** 2) / (tau ** 2 + se2)
                b_shrunk = w * b_i + (1.0 - w) * b_med
                b_shrunk = min(max(b_shrunk, SLOPE_MIN), SLOPE_MAX)
                # Re-centre on the data: the shrunk slope pivots at the
                # fitted points' centroid, not at x = 0.
                e_target = math.exp(my_i + b_shrunk * (x_target - mx_i))
                s_i = 0.3 if s_i is None else min(max(s_i, RESID_FLOOR), RESID_CAP)
                var = (s_i ** 2 + s_n ** 2) * infl + horizon_var + stale_var
            e_target = max(e_target, last_e[i])
            e_target = min(e_target, 0.9999 * n_target)
            med = clamp(100.0 * e_target / n_target)
            spread = min(max(math.sqrt(max(var, 0.0)), SPREAD_FLOOR), SPREAD_CAP)
        lg = logit(med)
        q = [round4(clamp(inv_logit(lg + z * spread))) for z in Z]
        q = sorted(q)
        rows.append({"id": trophy["id"], "q": q})
    return {"trophies": rows}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Every case file, in name order, so two runs write the same bytes.
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
