"""gta6-twitch-ccu-forecast-01: console rolling-midnight launch forecaster.

Forecasts Twitch viewers in the Grand Theft Auto VI category for hours 2..24
of release day (2026-11-19 UTC) from hour 1's print, using the launch curves
of past big games shipped in the case file (`comparables`).

Model (per forecast hour h, fitted at runtime from the case's comparables):
  log viewers(h) = a_h + b_h * log viewers(1)          (ordinary least squares)
  median(h)      = exp(a_h + b_h * log p1)
  q_tau(h)       = exp(median_log(h) + z_tau * sd_h * infl_h)
where sd_h is the regression residual sd and infl_h =
sqrt(1 + 1/n + (log p1 - mean log p1)^2 / Sxx) is the standard
prediction-interval inflation (slope/intercept uncertainty + extrapolation:
GTA VI's hour-1 print is expected at or above the largest training launch).

Training set: comparables with a complete release-day curve (hours 1..24,
viewers > 0) whose launch matches GTA VI's — console-only (no "PC" in
`platforms`) or a rolling midnight-local unlock ("midnight local" in
`unlock`). That excludes simultaneous-global-unlock launches (whose hour 1
is a pre-unlock trickle and whose log-ratio explodes) and early-access
contaminated curves, while keeping the big rolling-midnight launches
(ELDEN RING, Resident Evil Requiem, Call of Duty: Modern Warfare 4, ...).
The Grand Theft Auto VI pre-release comparable has no release-day hour 1
and is excluded by the completeness requirement. Fallbacks: if fewer than
5 matching curves, use every complete comparable; if fewer than 3, fall
back to the house persistence baseline.

Why log viewers on log hour-1, not a median log-ratio: the log-ratio
r(h) = log(v(h)/v(1)) is strongly scale-dependent (corr(log v1, r(21)) =
-0.94 over all comparables) because hour 1 is set by *when* a game
unlocked relative to UTC midnight, not by its size. The per-hour OLS
absorbs that (slopes 0.59-1.03) and leaves residual sd ~0.14-0.51, vs
~0.7-1.1 for a median-ratio over all comparables.

Validation (leave-one-out over 37 comparables + 4 practice launches,
truth for the practice cases fetched from SullyGnome's "Average viewers"
dataset and anchor-verified against each case's history_hourly tail):
  starter (persistence) practice loss 0.387, console-like LOO 0.344
  this model               practice loss 0.180, console-like LOO 0.184
  (pinball loss on the log scale; lower is better)

Python 3.13, standard library only, no network, deterministic.
"""

import json
import math
import os
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0, 0.674, 1.645]
SIGMA_FALLBACK = 0.15  # house baseline spread, used only if < 3 training curves


def round2(x: float) -> float | int:
    """x rounded to 2 decimals, half up (same convention as the starter)."""
    scaled = x * 100
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 100
    return int(value) if value.is_integer() else value


def _parse(t: str) -> datetime:
    return datetime.fromisoformat(t.replace("Z", "+00:00"))


def _is_console_only(comp: dict) -> bool:
    return "PC" not in str(comp.get("platforms", ""))


def _is_rolling(comp: dict) -> bool:
    return _is_console_only(comp) or "midnight local" in str(
        comp.get("unlock", "")).lower()


def _day_curve(comp: dict) -> dict | None:
    """Release-day hourly viewers {hour: viewers} for hours 1..24, or None
    when the curve is incomplete (e.g. the GTA VI pre-release comparable)."""
    try:
        ds = _parse(comp["day_start"])
    except Exception:
        return None
    m = {}
    for pt in comp.get("hourly") or []:
        try:
            t = _parse(pt["from"])
            v = pt["viewers"]
        except Exception:
            continue
        if v is None or v <= 0:
            continue
        dh = (t - ds).total_seconds() / 3600
        if dh == int(dh) and 1 <= dh + 1 <= 24:
            m[int(dh) + 1] = float(v)
    if len(m) < 24 or 1 not in m:
        return None
    return m


def _ols(xs, ys):
    """Least squares y ~ a + b*x; returns (a, b, residual_sd)."""
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    b = (sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx) if sxx > 0 else 0.0
    a = my - b * mx
    resid = [y - (a + b * x) for x, y in zip(xs, ys)]
    sd = math.sqrt(sum(r * r for r in resid) / max(1, n - 2)) if n > 2 else 0.5
    return a, b, sd, mx, sxx


def _persistence(p1: float, h: int) -> list:
    return [round2(p1 * math.exp(z * SIGMA_FALLBACK * math.sqrt(h - 1))) for z in Z]


def forecast_case(case: dict, name: str) -> dict:
    first = next((k for k in case["known"] if k["hour"] == 1), None)
    if first is None:
        raise ValueError(f"{name} has no hour-1 print")
    p1 = max(1.0, float(first["viewers"]))
    lv1 = math.log(p1)

    curves = []
    for comp in case.get("comparables") or []:
        m = _day_curve(comp)
        if m is not None:
            curves.append((comp, m))
    rolling = [(c, m) for c, m in curves if _is_rolling(c)]
    train = rolling if len(rolling) >= 5 else curves
    if len(train) < 3:
        return {"hours": [{"hour": row["hour"], "q": _persistence(p1, row["hour"])}
                         for row in case["forecast"]]}

    hours = []
    for row in case["forecast"]:
        h = row["hour"]
        xs = [math.log(m[1]) for _, m in train]
        ys = [math.log(m[h]) for _, m in train]
        a, b, sd, mx, sxx = _ols(xs, ys)
        n = len(xs)
        infl = math.sqrt(1.0 + 1.0 / n + (lv1 - mx) ** 2 / sxx) if sxx > 0 else 1.0
        mu = a + b * lv1
        spread = sd * infl
        qs = []
        prev = 0.0
        for z in Z:
            q = round2(math.exp(mu + z * spread))
            q = max(q, 1.0)
            q = min(q, 1e9)
            if q < prev:  # keep the row non-decreasing after rounding
                q = prev
            qs.append(q)
            prev = q
        hours.append({"hour": h, "q": qs})
    return {"hours": hours}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
