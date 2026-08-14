# Walmart DemandLens — Production Architecture

## 1. Application layers

### Presentation
`streamlit_app.py`

Responsible for:

- navigation
- responsive dashboard UI
- forecast controls
- execution progress
- charts
- model comparison
- insights
- report downloads
- system status

There is intentionally no upload/drag-and-drop component.

### Domain/data
`demandlens/walmart/`

Responsible for:

- Walmart file discovery
- schema validation
- loading
- profiling
- data-quality checks
- bundled future calendar

### Forecasting
`demandlens/forecasting/`

Responsible for:

- time-series aggregation
- lag/rolling features
- SARIMA
- Prophet
- LightGBM
- metrics
- chronological backtesting

### Intelligence
`demandlens/insights/`

Converts numerical forecast outputs into:

- trend interpretation
- forecast change
- holiday effect
- variability/confidence indicator
- prioritized actions

### Reporting
`demandlens/reports/`

Creates:

- CSV forecast export
- visual PDF decision report

Matplotlib uses `Agg` to prevent GUI/window creation during Streamlit execution.

## 2. Execution lifecycle

```text
START
  |
  v
Validate Walmart files
  |
  v
Load train.csv
  |
  v
Filter Store/Dept
  |
  v
Aggregate weekly series
  |
  v
Chronological holdout
  |
  +--> SARIMA ----+
  |               |
  +--> Prophet ---+--> Metrics --> WMAE ranking
  |               |
  +--> LightGBM --+
                      |
                      v
               Winning model
                      |
                      v
             Full-history training
                      |
                      v
              Future prediction
                      |
                      v
               AI interpretation
                      |
              +-------+-------+
              |               |
              v               v
             PDF             CSV
```

## 3. Progress contract

Progress values represent **pipeline state boundaries**.

They do not represent fabricated elapsed-time estimates.

Model events:

- `started`
- `fitted`
- `predicted`
- `completed`
- `failed`

The UI displays the timestamps attached to those events.

## 4. Model selection

Successful validation results are ranked:

```python
winner = min(successful_models, key=lambda model: model["wmae"])
```

This avoids hard-coding LightGBM, Prophet or SARIMA as the winner.

## 5. Production report contract

The PDF and CSV are generated from the same forecast payload that powers the Streamlit result.

This prevents a common reporting problem where the dashboard and downloaded report show different calculations.

## 6. Appearance

No application-level theme switch exists.

Custom styling uses:

```css
@media (prefers-color-scheme: dark)
```

so the custom DemandLens surfaces respond to the user's system/browser preference.

## 7. Deployment

Primary local:

```text
streamlit run streamlit_app.py
```

Container:

```text
Docker
  |
  +-- Python 3.12
  +-- Streamlit
  +-- Forecasting libraries
  +-- Bundled Walmart data
```
