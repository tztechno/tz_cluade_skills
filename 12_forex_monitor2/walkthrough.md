# USD/JPY Forex Tracker Walkthrough

I have completed the implementation of the USD/JPY Forex tracker according to the approved plan.

## Completed Features

### Backend (FastAPI)
- `main.py` is fully implemented and running locally on port 8000.
- A background task fetches the `USDJPY=X` Forex data from Yahoo Finance (`yfinance`) every minute.
- The data is correctly saved to `data/forex_data.csv`.
- The `/api/data` endpoint parses the CSV and returns the latest 60 records (1 hour) as requested.

### Frontend (HTML/JS)
- `static/index.html` offers a premium, responsive dark-mode UI.
- The chart renders the past 1-hour USD/JPY closing prices dynamically using **Chart.js**.
- Replaced the chart time adapter to use `date-fns` for better compatibility.
- Interactive inputs for **Upper Limit** and **Lower Limit** have been added. They auto-save locally so you don't lose them on refresh.
- If the Forex rate exceeds the Upper Limit or falls below the Lower Limit, the line segments will turn **RED**. Horizontal red dashed lines indicate the configured limits.

## How to Test

Since the server is running in the background, you can simply open your browser and go to:

👉 **[http://127.0.0.1:8000](http://127.0.0.1:8000)**

1. Wait for the chart to populate with data (it fetches 1 hour of history immediately).
2. Look at the current price and input an **Upper Limit** slightly below it or a **Lower Limit** slightly above it to instantly see the red line indications!

The application will now automatically update the CSV file and the frontend chart every minute.
