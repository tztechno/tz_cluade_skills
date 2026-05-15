# USD/JPY Forex Tracker

This project will create a web application that fetches past 1-hour USD/JPY Forex data, automatically updates it every minute, saves the data to a CSV file, and displays it on a dynamic graph. 

## User Review Required

> [!WARNING]
> You mentioned "backをfastaiで" (backend with fastai). `fastai` is a deep learning library. I strongly suspect you meant **FastAPI**, which is a modern, fast web framework for Python. This plan assumes the use of **FastAPI**. Please confirm if this is correct.

> [!NOTE]
> I propose using the `yfinance` Python library to fetch the USD/JPY data, as it is free and does not require an API key. 

## Open Questions

1. **Limit persistence:** Should the Upper Limit and Lower Limit be saved on the backend so they are shared across all users, or is it okay if they are just controlled in the frontend browser by the user viewing the page? (The plan currently assumes frontend control).
2. **"赤線表示にする" (Display in red line):** My plan is to display the Upper and Lower limit threshold lines as **red horizontal lines**, AND change the color of the **data graph line to red** when it goes outside these limits. Does this match your expectation?

## Proposed Changes

### Backend (FastAPI)

#### [NEW] `requirements.txt`
Dependencies including `fastapi`, `uvicorn`, `yfinance`, `pandas`.

#### [NEW] `main.py`
- Setup a FastAPI application.
- Implement an `asyncio` background task that runs every 1 minute to fetch the latest `USDJPY=X` data using `yfinance`.
- On startup, it will fetch the last 1 hour of data.
- Save and append the fetched `(timestamp, price)` data to a CSV file (`data/forex_data.csv`).
- Create an API endpoint `/api/data` to serve the recent data to the frontend in JSON format.
- Create an endpoint to serve the static `index.html`.

### Frontend (HTML/JS)

#### [NEW] `static/index.html`
- A premium, modern dark-themed web page.
- Input fields to set the **Upper Limit** and **Lower Limit**.
- A chart using **Chart.js**.
- JavaScript logic to fetch data from `/api/data` every minute.
- Chart.js configuration to draw horizontal limit lines.
- Segment styling in Chart.js to change the line color to **RED** if the value exceeds the Upper Limit or falls below the Lower Limit.

## Verification Plan

### Automated / Backend Verification
- Run the FastAPI server locally.
- Verify `data/forex_data.csv` is created and updated every minute with valid timestamps and USD/JPY prices.

### Manual Verification
- Open the web application in a browser.
- Verify the graph displays the last hour of data.
- Input arbitrary Upper and Lower limits and verify that the red lines appear.
- Verify that if the price crosses the limits, the graph visually indicates it in red.
