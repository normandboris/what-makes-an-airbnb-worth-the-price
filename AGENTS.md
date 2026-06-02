# AGENTS.md

## Cursor Cloud specific instructions

### What this repo is

A single Jupyter notebook (`airbnb_predictor.ipynb`) that predicts NYC Airbnb nightly prices (Linear Regression + Random Forest) and writes `nyc_airbnb_choropleth.html`. There is no API, database, Docker stack, or automated test/lint configuration.

### Dependencies

Install per `README.md`, plus Jupyter tooling for headless runs. See `README.md` for the base `pip install` line.

**Important:** Pin **pandas 2.x** (`pandas>=2.0,<3`). The notebook only cleans `price` when `dtype == 'O'`. Pandas 3 uses string dtype (`str`), so prices stay as strings and groupby/median fails.

User-local installs put CLIs under `~/.local/bin`. Ensure `export PATH="$HOME/.local/bin:$PATH"` before running `jupyter` or `jupyter-nbconvert`.

### Data files (not in git)

Place next to the notebook in the repo root:

| File | Source |
|------|--------|
| `listings.csv.gz` | Inside Airbnb NYC **detailed** listings. Prefer a snapshot with non-empty `price` values. As of setup, `2026-02-13` had all-null prices; `2025-06-04` works: `http://data.insideairbnb.com/united-states/ny/new-york-city/2025-06-04/data/listings.csv.gz` |
| `neighbourhoods.geojson` | Inside Airbnb **visualisations** path (the `data/` path often returns 403): `https://data.insideairbnb.com/united-states/ny/new-york-city/2026-02-13/visualisations/neighbourhoods.geojson` |

The README NYC Open Data GeoJSON URL may return HTTP 400; use the Inside Airbnb visualisations link above.

### Run / verify

- **Interactive:** `jupyter notebook airbnb_predictor.ipynb` → Kernel → Restart & Run All (~30s for Random Forest).
- **Headless (CI-style):** `MPLBACKEND=Agg jupyter nbconvert --to notebook --execute airbnb_predictor.ipynb --ExecutePreprocessor.timeout=600`
- **Map output:** open `nyc_airbnb_choropleth.html` in a browser (Leaflet tiles load from CDNs).

### Lint and tests

None configured. Validation is notebook execution and inspecting model metrics / regenerated HTML.
