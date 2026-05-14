---
name: fastapi-realtime-dashboard
description: >
  Build a realtime financial / data dashboard web app with a FastAPI backend and a
  single-file HTML frontend. Use this skill whenever the user wants to create a web
  app that fetches live market data (forex, stocks, crypto, etc.), visualises it with
  charts and custom SVG diagrams, and auto-refreshes on a timer. Also trigger for any
  FastAPI + HTML project that needs: moving-average calculations, vector/geometric
  visualisations, Chart.js integration, dark-theme terminal-style UI, or a demo-mode
  fallback when external data is unavailable. Even if the user just says "make a web
  app that shows live prices" or "visualise currency strength" — use this skill.
---

# FastAPI Realtime Dashboard Skill

## What this skill produces

A self-contained web app with:
- **`main.py`** — FastAPI backend: data fetching, calculations, JSON API
- **`static/index.html`** — Single-file frontend: Chart.js charts + SVG diagram + auto-refresh
- **`requirements.txt`** — Python dependencies
- **`start.sh`** — One-command launcher
- **`README.md`** — Setup and architecture notes

---

## Project structure

```
project/
├── main.py
├── requirements.txt
├── start.sh
├── README.md
└── static/
    └── index.html
```

**Critical**: the `static/` directory must exist before starting the server, or
`StaticFiles` raises `RuntimeError: Directory 'static' does not exist`.
Always remind the user to run `mkdir -p static` if they clone/copy the files manually.

---

## Backend — `main.py` patterns

### Boilerplate
```python
from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def root():
    return HTMLResponse(Path("static/index.html").read_text(encoding="utf-8"))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
```

### Safe float helper (always include)
```python
import math

def safe_float(v):
    try:
        f = float(v)
        return None if math.isnan(f) or math.isinf(f) else round(f, 6)
    except Exception:
        return None

def to_list(series):          # pandas Series → JSON-safe list
    return [safe_float(v) for v in series]
```

### yfinance data fetch
```python
import yfinance as yf
import pandas as pd

def get_close(df: pd.DataFrame) -> pd.Series:
    """Handle yfinance MultiIndex columns safely."""
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df["Close"].squeeze()

# Fetch 2 days of 1-min bars, then take the last 60
raw = yf.download("USDJPY=X", period="2d", interval="1m", progress=False)
series = get_close(raw).dropna().tail(60)
```

**Always use `period="2d"`** for 1-minute data — `period="1d"` sometimes returns
fewer than 60 bars on short trading days or around market open.

### Demo/fallback mode
Always add a `?demo=true` query parameter that returns synthetic random-walk data.
This lets the user verify the UI without network access or during market closure.

```python
import random
from datetime import datetime

def make_demo_series(start: float, drift: float, sigma: float, n: int = 60) -> pd.Series:
    rng = random.Random()
    vals = [start]
    for _ in range(n - 1):
        vals.append(vals[-1] * (1 + rng.gauss(drift, sigma)))
    now = datetime.now().replace(second=0, microsecond=0)
    return pd.Series(vals, index=pd.date_range(end=now, periods=n, freq="1min"))

@app.get("/api/data")
async def get_data(demo: bool = Query(False)):
    if demo:
        return build_response(
            make_demo_series(155.0, 0.00003, 0.0003),  # USDJPY
            make_demo_series(1.085, -0.00002, 0.0002),  # EURUSD
        )
    # ... real fetch
```

### Error response convention
```python
from fastapi.responses import JSONResponse

# Success
return {"ok": True, "timestamps": [...], "data": {...}}

# Error
return JSONResponse({"ok": False, "error": str(e)}, status_code=500)
```

Always check `d.ok` in the frontend before rendering.

---

## Power / strength vector calculation (multi-currency)

For N currencies derived from M cross rates, decompose each cross rate's
percentage change into additive contributions per currency, then project
onto the vertices of a regular polygon.

### Example: USD / JPY / EUR from USDJPY + EURUSD

```python
def calculate_vectors(usdjpy: pd.Series, eurusd: pd.Series) -> dict:
    """
    For window in {5, 20, 60}:
      change_N = (MA_N[-1] - close[-N]) / close[-N] * 100

      USD_power =  Δ(USDJPY) − Δ(EURUSD)
      JPY_power = −Δ(USDJPY)
      EUR_power =  Δ(EURUSD)

    Triangle vertices (centroid = origin):
      USD: (0,      −1   )  ← top
      JPY: (−√3/2, +0.5 )  ← bottom-left
      EUR: (+√3/2, +0.5 )  ← bottom-right

    Result vector = Σ power_i × vertex_i
    """
    import math
    sqrt3_2 = math.sqrt(3) / 2
    results = {}

    for label, window in [("ma5", 5), ("ma20", 20), ("ma60", 60)]:
        n = len(usdjpy)
        if n < window:
            results[label] = dict(usd=0.0, jpy=0.0, eur=0.0, vec_x=0.0, vec_y=0.0)
            continue

        ma_u = float(usdjpy.rolling(window).mean().iloc[-1])
        ma_e = float(eurusd.rolling(window).mean().iloc[-1])
        ref_idx = max(0, n - window - 1)
        ref_u = float(usdjpy.iloc[ref_idx]) or ma_u
        ref_e = float(eurusd.iloc[ref_idx]) or ma_e

        uj_chg = (ma_u - ref_u) / ref_u * 100
        eu_chg = (ma_e - ref_e) / ref_e * 100

        usd_p = uj_chg - eu_chg
        jpy_p = -uj_chg
        eur_p = eu_chg

        vx = jpy_p * (-sqrt3_2) + eur_p * sqrt3_2
        vy = usd_p * (-1.0)     + jpy_p * 0.5 + eur_p * 0.5

        results[label] = dict(
            usd=round(usd_p, 5), jpy=round(jpy_p, 5), eur=round(eur_p, 5),
            vec_x=round(vx, 5),  vec_y=round(vy, 5),
        )
    return results
```

---

## Frontend — `static/index.html` patterns

### Layout

```
┌─────────────────┬─────────────────┐
│                 │  Chart A        │
│  SVG Diagram    ├─────────────────┤
│  (vector / tri) │  Chart B        │
└─────────────────┴─────────────────┘
```

CSS grid split:
```css
.main-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;   /* 50/50 — adjust as needed */
  min-height: calc(100vh - 65px);   /* 65px = header height */
}
```

### Design system (dark terminal theme)

```css
:root {
  --bg:      #050a0f;
  --bg2:     #080f18;
  --bg3:     #0c1620;
  --border:  #0d2a3d;
  --text:    #8ab4cc;
  --text-hi: #c8e8f8;
  --accent:  #00d4ff;   /* cyan  — primary highlight */
  --green:   #00ff9d;   /* up / positive */
  --yellow:  #ffe066;   /* warning / mid */
  --red:     #ff4d6d;   /* down / negative */
  --mono:    'Share Tech Mono', monospace;
  --head:    'Rajdhani', sans-serif;
}
```

Google Fonts import:
```html
<link href="https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Rajdhani:wght@300;400;600;700&display=swap" rel="stylesheet">
```

Optional scanline overlay (adds CRT feel at zero cost):
```css
body::before {
  content: '';
  position: fixed; inset: 0;
  background: repeating-linear-gradient(
    0deg, transparent, transparent 2px,
    rgba(0,0,0,0.07) 2px, rgba(0,0,0,0.07) 4px
  );
  pointer-events: none;
  z-index: 999;
}
```

### Chart.js integration

CDN:
```html
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
```

Minimal dark config:
```javascript
Chart.defaults.color = '#2d566e';
Chart.defaults.borderColor = '#0a1e2c';

const cfg = {
  type: 'line',
  data: { labels: [], datasets: [
    { label: 'Close', data: [], borderColor: '#00d4ff', borderWidth: 1.5,
      pointRadius: 0, tension: 0.3, spanGaps: false },
    { label: 'MA5',  data: [], borderColor: '#00ff9d', borderWidth: 1.8,
      pointRadius: 0, spanGaps: false },
    { label: 'MA20', data: [], borderColor: '#ffe066', borderWidth: 1.8,
      pointRadius: 0, spanGaps: false },
    { label: 'MA60', data: [], borderColor: '#ff8c42', borderWidth: 2.0,
      pointRadius: 0, spanGaps: false },
  ]},
  options: {
    responsive: true, maintainAspectRatio: false,
    animation: { duration: 300 },
    plugins: {
      legend: { labels: { font: { family: "'Share Tech Mono'", size: 9 },
                          boxWidth: 12, padding: 6 } },
    },
    scales: {
      x: { grid: { color: '#0a1e2c' },
           ticks: { font: { family: "'Share Tech Mono'", size: 8 },
                    maxTicksLimit: 6, maxRotation: 0 } },
      y: { position: 'right', grid: { color: '#0a1e2c' },
           ticks: { font: { family: "'Share Tech Mono'", size: 9 },
                    maxTicksLimit: 6 } },
    }
  }
};
```

Update without full redraw:
```javascript
function updateChart(chart, labels, close, ma5, ma20, ma60) {
  chart.data.labels = labels;
  chart.data.datasets[0].data = close;
  chart.data.datasets[1].data = ma5;
  chart.data.datasets[2].data = ma20;
  chart.data.datasets[3].data = ma60;
  chart.update('none');   // 'none' skips animation for live updates
}
```

### SVG Vector Diagram

Equilateral triangle geometry (circumradius `R`, centroid at `(CX, CY)`):
```javascript
const CX = 240, CY = 228, R = 175;

const VERTS = {
  usd: [CX,                       CY - R],
  jpy: [CX - R * Math.sqrt(3)/2,  CY + R/2],
  eur: [CX + R * Math.sqrt(3)/2,  CY + R/2],
};
```

Auto-scale vectors to fit inside ~110 px:
```javascript
function computeScale(vecs) {
  const mags = Object.values(vecs).map(v => Math.hypot(v.vec_x, v.vec_y));
  const maxMag = Math.max(...mags, 0.0001);
  return Math.min(110 / maxMag, 1200);   // cap prevents invisible tiny markets
}

function drawVec(svgLineId, vecX, vecY, scale) {
  const el = document.getElementById(svgLineId);
  el.setAttribute('x1', CX);
  el.setAttribute('y1', CY);
  el.setAttribute('x2', CX + vecX * scale);
  el.setAttribute('y2', CY + vecY * scale);
}
```

SVG `<defs>` for arrowhead markers (one per MA colour):
```xml
<marker id="arr5" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto">
  <polygon points="0 0, 7 3.5, 0 7" fill="#00ff9d"/>
</marker>
```

### Auto-refresh with countdown

```javascript
let countdown = 60;

function tick() {
  countdown--;
  document.getElementById('countdown').textContent = countdown + 's';
  if (countdown <= 0) { countdown = 60; fetchData(); }
}

fetchData();
setInterval(tick, 1000);
```

### Error banner with demo fallback button

```javascript
if (!d.ok) {
  const eb = document.getElementById('error-banner');
  eb.style.display = 'block';
  eb.innerHTML = '⚠ ' + d.error +
    ' &nbsp;<button onclick="toggleDemo()">Switch to Demo</button>';
}
```

---

## requirements.txt

```
fastapi>=0.110.0
uvicorn[standard]>=0.29.0
yfinance>=0.2.40
pandas>=2.0.0
numpy>=1.26.0
```

---

## start.sh

```bash
#!/bin/bash
cd "$(dirname "$0")"
pip install -r requirements.txt --quiet --break-system-packages
python main.py
```

---

## Common pitfalls

| Problem | Cause | Fix |
|---|---|---|
| `RuntimeError: Directory 'static' does not exist` | `static/` folder missing | `mkdir -p static` |
| Empty DataFrame from yfinance | `period="1d"` too short | Use `period="2d"`, take `.tail(60)` |
| MultiIndex columns from yfinance | Multiple tickers or newer yfinance | Call `get_close()` helper above |
| NaN serialisation error | pandas NaN not JSON-safe | Use `safe_float()` / `to_list()` helpers |
| Vectors invisible (all zero) | MA window > available bars | Guard with `if n < window: skip` |
| Charts look squashed | `maintainAspectRatio: true` | Set `maintainAspectRatio: false` + wrap canvas in flex container |

---

## Checklist before delivering the app

- [ ] `static/` directory referenced correctly (relative to where `python main.py` is run)
- [ ] `safe_float` / `to_list` applied to all pandas output before JSON serialisation
- [ ] `?demo=true` endpoint returns plausible synthetic data
- [ ] Frontend checks `d.ok` before rendering
- [ ] Error banner shows helpful message + demo toggle button
- [ ] `chart.update('none')` used for live refreshes (no animation stutter)
- [ ] `computeScale` applied so vectors are always visible regardless of market volatility
- [ ] Layout grid columns set to desired ratio (default `1fr 1fr` = 50/50)
