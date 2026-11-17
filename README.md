# macro-event-vol

Course project: selling macro event-day volatility with capped risk (base
strategy), plus a "Fed focus" overlay that sizes each event by how much the
release matters for policy (research component).

## Setup

1. Clone the repo and create an environment:

   ```
   python -m venv .venv
   source .venv/bin/activate        # Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. Copy `.env.example` to `.env` and fill in:
   - `WRDS_USERNAME`: your own WRDS username. The first connection asks for
     your password and offers to save it in a local `.pgpass` file.
   - `FRED_API_KEY`: free key from
     https://fred.stlouisfed.org/docs/api/api_key.html
   - `DATA_DIR`: path to your local sync of the shared data folder. Leave it
     blank to use `./data`.

3. Run the pipeline from the repo root:

   ```
   python run.py              # all steps
   python run.py --step data  # one step: data, events, options, signal, backtest
   python run.py --step events --refresh  # re-pull instead of reading cached files
   ```

## Data

No data is committed to git. OptionMetrics and Bloomberg data are licensed.

| Source | Who pulls it | How |
|---|---|---|
| WRDS (OptionMetrics) | anyone | `python run.py --step data`, using your own login |
| FRED / ALFRED | anyone | same step; public |
| Bloomberg | whoever is at the terminal | export the series listed in `config.yaml` to `DATA_DIR/raw/bloomberg/` |

`raw/` holds files exactly as pulled and is never edited by hand.
`processed/` holds everything the code builds from them.

## Layout

```
config.yaml     events, series IDs, strategy parameters
src/data.py     pulls from WRDS and FRED, reads Bloomberg exports
src/events.py   event calendar and surprises
src/options.py  contract selection and pricing of the option structure
src/signal.py   Fed focus score
src/backtest.py base strategy, overlay, metrics
run.py          runs the pipeline end to end
notebooks/      exploration and figures (prefix files with your initials)
```

## Workflow

- Work on a branch and merge to `main` by pull request.
- Change parameters in `config.yaml`, not in code.
- Fix the event list, the series in each composite and the sizing rule in
  `config.yaml` before looking at backtest results.
