# GTA VI Trophy Rarity Forecast — gta6-trophy-rarity-forecast-01

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/0x-zeze/gta6-trophy-rarity-forecast-01/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/0x-zeze/gta6-trophy-rarity-forecast-01/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Forecast how many GTA VI players earn each trophy in week one: for each of
the game's 55 trophies, five quantiles (5%, 25%, 50%, 75%, 95%) of its earn
rate at the target, from Exophase's launch-night reads of the PS5 trophy page.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Python 3.13, standard library only, no install step, no network. In the
grader's container it reads the case at `/input/<id>.in` and writes
`/output/<id>.out`. Deterministic: two runs write the same bytes. Runs in
well under the 8 s cap per case.

## Model: earners over players

An earn rate is a ratio — `rate = 100 * earners / players_tracked` — and both
counts only move when Exophase refreshes the page, so a case's history is a
step function: long flat stretches, then a jump. The forecaster works on the
counts, not the rates:

- It collapses the history into distinct states (consecutive identical reads
  are one state, kept at the time it first appeared).
- It fits, over the last 24 hours of distinct states, a least-squares line of
  `ln(players_tracked)` against `ln(1 + hours since launch)`, and one line per
  trophy of `ln(earners)` against the same clock (earners recovered as
  `rate * players_tracked / 100`).
- Each trophy's slope is shrunk toward the cross-trophy median slope by
  empirical-Bayes weights: a slope the data pin down precisely is trusted
  almost fully, a noisy one is pulled hard toward the median. The standard
  error is inflated by the page's rounding (rates print to 2 decimals, so a
  trophy at 0.01% carries ~50% relative noise in its recovered earners).
- It reads both lines at the target, divides, and clamps the rate to
  `[0.01, 99.99]`. Earners never decrease; a trophy's forecast earners never
  exceed the forecast players tracked.

Two cases the trend fit cannot see need priors, both stated in the
competition's own rules:

- **Flat history** (launch night: the page sat unchanged at 638 players
  tracked for hours) has no trend to fit, yet a refresh is coming — the rules
  record the first refresh taking tracked players 638 → 3,868. The forecaster
  therefore grows tracked players by a power law in log time since launch,
  `N(T) = N_last * (1 + h_T) ** 1.05`, which reproduces that first refresh
  (638 × 5.5^1.05 ≈ 3,820, against the recorded 3,868), and grows each
  trophy's earners by a small category prior (early-story trophies faster
  than late/grind ones), so the forecast rate is the diluted rate, not the
  stale one.
- A trophy already at ~100% (the opening mission, earned by essentially every
  tracked player) stays there. A trophy with no earners anywhere stays at the
  0.01% floor with a wide spread.

Spreads combine the residual scatter of the earners and players fits
(inflated by how far the target sits beyond the fitted clock range), a base
term growing with the horizon in days, and a term for a stale last state,
floored so the 90% band stays honest and capped so one wild trophy cannot
blow out the mean loss.

## Validation

Scored exactly as the contest scores (mean pinball loss on the logit scale;
lower is better):

| Test | This model | House persistence baseline |
|---|---|---|
| Backtest B: flat launch-night history → live Exophase page 18.8 h later (2026-11-19T22:17Z, tracked players 638 → 10,000+) | **0.20** | 2.31 |
| Synthetic final-like case: simulated multi-refresh history over 59.5 h, target 4.5 days out | **0.10** | 0.36 |

(The synthetic world uses the same power-law family as the model, so that
number is optimistic; the real world will deviate, which is what the spreads
are for.)

## Files

- `main.py` — the whole entry: reads the case, fits the earners-over-players
  model, writes the 55-row forecast.
- `LICENSE` — MIT.
- `.makefun/owner.json` — the payout wallet.

---

<sub>Entered on [make.fun](https://make.fun/?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
