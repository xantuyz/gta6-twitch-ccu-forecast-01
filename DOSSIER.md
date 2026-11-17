# DOSSIER — gta6-twitch-ccu-forecast-01

## Task

Forecast Twitch concurrent viewers in the "Grand Theft Auto VI" category
for hours 2..24 of release day 2026-11-19 (UTC), as quantiles
(0.05/0.25/0.5/0.75/0.95) per hour, from hour 1's print. Score: mean
pinball loss on the log scale. One final case (h00); practice: 4 past
console launches (p00 Marvel's Wolverine, p01 Red Dead Redemption 2,
p02 God of War Ragnarök, p03 Marvel's Spider-Man 2).

## What the data showed

- Hour 1 is set by *unlock timing*, not game size. Games with a
  simultaneous global unlock later in the UTC day (e.g. Hollow Knight:
  Silksong, hour 1 = 164 viewers; Black Myth: Wukong, 4,941) have
  log-ratios r(h) = ln(v(h)/v(1)) up to +7. Games whose PC version
  unlocked the evening before (ELDEN RING, Cyberpunk 2077) have hour 1
  already at the peak. Across all 37 comparables, corr(log v1, r(21)) =
  -0.94, and the per-hour sd of r(h) is 0.7–2.2 — a median-ratio forecast
  over all comparables is hopeless (LOO loss 0.79).
- Console-only rolling-midnight launches (8 comparables + 4 practice
  launches) are a different, much tighter population: median r rises
  from -0.55 at hour 4 (pre-US-unlock trough) to +0.89 at hour 20
  (US-evening peak), per-hour sd 0.12–0.47. GTA VI unlocks at rolling
  local midnight (CET 23:00Z Nov 18, UK 00:00Z, US East 05:00Z,
  US Pacific 08:00Z), so it belongs to this population.
- Hour-1 channels add nothing beyond hour-1 viewers (corr with the
  ratio is driven by the same scale effect); the target's own
  pre-release history (24 h peak) correlates 0.94 with v1 but only
  0.12 with the release-day peak — no extra signal.

## Final model (in main.py, fitted at runtime from the case file)

- Training set: comparables with a complete release-day curve
  (hours 1..24, viewers > 0) that are console-only (no "PC" in
  `platforms`) or have a rolling midnight-local unlock ("midnight
  local" in `unlock`). n = 20 in the final case (16 comparables +
  4 practice launches). The GTA VI pre-release comparable has no
  release-day hour 1 and is excluded by completeness. Fallbacks:
  < 5 matching curves → all complete comparables; < 3 → house
  persistence baseline.
- Per hour h: OLS log v(h) ~ a_h + b_h · log v1. Fitted slopes
  0.59–1.03 (late hours ~0.6: strong regression to the mean — the
  peak is only weakly tied to hour 1). Median(h) = exp(a_h + b_h·log p1).
- Spread: residual sd × sqrt(1 + 1/n + (log p1 − mean log p1)²/Sxx)
  (prediction interval: slope/intercept uncertainty + extrapolation —
  GTA VI's hour-1 print is expected at/above the largest training
  launch, 875k). Quantiles via normal z = (−1.645, −0.674, 0, 0.674, 1.645).
- All in log-ratio form: nothing depends on absolute scale, so the
  model extrapolates to a GTA VI hour-1 print far above the training
  range. An explicit extrapolation test (predict the 4 launches with
  v1 > 300k from a training set with all of them removed) favoured
  plain OLS (0.228) over a slope-→1-beyond-range variant (0.288) and
  over median-ratio (0.413).
- Example: at a hypothetical hour-1 print of 1,000,000, the median
  curve dips to ~518k at hour 12 and peaks at ~944k at hour 19
  (90% interval at hour 20: ~438k–1.94M).

## Validation

Ground truth for the 4 practice cases was fetched from SullyGnome's
"Average viewers" dataset (4 requests, one month per game, per
input/NOTICE.txt) and anchor-verified: each case's known hour-1 value
appears exactly once in its month's dataset, and the preceding values
match the case file's `history_hourly` tail exactly.

| Method | Practice (4 cases) | Console-like LOO (12) | LOO all (41) | 90% coverage |
|---|---|---|---|---|
| House baseline (starter) | 0.387 | 0.344 | 0.877 | 0.639 |
| Median ratio, all comparables | 0.396 | 0.385 | 0.755 | 0.875 |
| Median ratio, console-only | 0.251 | 0.231 | 0.811 | 0.528 |
| OLS all comparables | 0.243 | 0.313 | 0.420 | 0.919 |
| **This model (OLS, rolling set + pred. interval)** | **0.180** | **0.186** | **0.549** | **0.891 (practice) / 0.702 (LOO)** |
| — per practice case | p00 0.249, p01 0.146, p02 0.128, p03 0.197 | | | |

LOO = leave-one-out over 37 comparables + 4 practice launches, each
target forecast from the other 40 curves (mimicking the final's 42
comparables), scored by the shipped `main.py` run as a subprocess.
The starter's practice loss reproduces the RULES figure (0.387),
validating both the pinball implementation and the fetched truth.

Format validation: `main.py` run once per practice case (grader-style,
one `.in` per run): all outputs valid — exactly 23 rows, hours 2..24,
5 non-decreasing quantiles each, all in (0, 1e9], files ~1.5 KB
(< 65,536). Runtime ~0.10–0.17 s per case (limit 8 s). Two runs
byte-identical (deterministic). Edge cases (no comparables; only the
GTA VI pre-release comparable) fall back to persistence and still
produce valid output.

## Limitations

- The final is a single case; LOO says ~0.18–0.23 is the expected loss
  for a launch of GTA VI's size, well below the house baseline (~0.39
  on practice) but above the current practice-leaderboard top
  (0.122) — the top practice entries may be tuned harder to those 4
  cases, which does not transfer to the final.
- Biggest risk: GTA VI's scale. No training launch had an hour-1
  print near what GTA VI may post; the OLS slope < 1 caps the
  predicted peak near ~1M viewers at a 1M hour-1 print, while a
  median-ratio model would say ~2.4M. The extrapolation test above
  supports the OLS cap, but if GTA VI breaks every previous peak by
  a wide margin, the evening hours will be under-forecast.
- Marvel's Wolverine (smallest practice hour-1, 23,357; peak 212,243,
  9.1×) is the worst practice case (0.249): small-hour-1 launches
  have the widest ratio dispersion, and nothing in the case data
  predicted its 9× rise.
- The final's hour-1 print is a mean of 60 minute reads (precise);
  practice hour-1 is a single SullyGnome hourly average (noisier),
  so practice loss slightly overstates final-case noise.
