# GTA VI Twitch CCU Forecast

Forecasts how many people watch Grand Theft Auto VI live on Twitch in each
hour of its release day (2026-11-19 UTC), as five quantiles per hour, from
the first hour's viewer count. An entry for the make.fun
**gta6-twitch-ccu-forecast-01** bounty ($250, top 20).

## How it works

`main.py` (Python 3.13, standard library only) reads the case file
(`/input/<id>.in`) and, for each hour 2..24, fits a per-hour regression of
log viewers on log hour-1 viewers over past big game launches shipped in
the case (`comparables`) — restricted to launches like GTA VI's: console
releases with a rolling midnight-local unlock. The median is the fitted
value at the case's hour-1 print; the spread is the residual standard
deviation widened by the standard prediction-interval factor (which grows
when the hour-1 print extrapolates beyond the training launches). See
[DOSSIER.md](DOSSIER.md) for the full strategy and validation numbers.

## Run it

```sh
mkdir -p out
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

This reads every `<id>.in` in `input/` and writes `<id>.out` to `out/`:
`{"hours": [{"hour": 2, "q": [q05, q25, q50, q75, q95]}, ...]}` — 23 rows,
hours 2..24, deterministic, ~0.1 s per case, no network.

## Competition

- Bounty: https://make.fun/c/gta6-twitch-ccu-forecast-01
- Score: mean pinball loss on the log scale over every hour and quantile
  (levels 0.05, 0.25, 0.5, 0.75, 0.95); lower is better.
- Practice loss (4 past launches, ground truth from SullyGnome): **0.180**
  vs house baseline **0.387**.

## License

MIT — see [LICENSE](LICENSE).
