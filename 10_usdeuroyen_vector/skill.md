Excellent choice. Having this in English is standard for technical documentation and makes it a great addition to a GitHub repository or a portfolio. Here is the translated `skill.md`.

---

# `skill.md`

# Skill Document: Real-Time Currency Strength Analysis Application Development

This document records the design and implementation methods of an application that visualizes moving averages of multiple currency pairs and currency power balance (via vector synthesis) in real time using a combination of **FastAPI (Python)** and **Plotly (JavaScript)**.

---

## 1. System Architecture

* **Backend: FastAPI (Python)**
* `GET /history`: Provides a bulk dataset for the past hour (60 data points) upon initial startup.
* `WebSocket /ws`: Delivers real-time data updates every minute via push notifications.


* **Frontend: Plotly.js + Vanilla JS**
* Initialization via asynchronous `fetch` requests followed by continuous data plotting through WebSockets.


* **Data Structure**
* JSON format is utilized to manage time series, raw prices, moving averages (MA), and synthesized vector coordinates in a unified manner.



---

## 2. Implementation Key Points

### A. Backend: Data Continuity and Integrity

* **Initialization Logic**: Uses `datetime` and `timedelta` to generate and retrieve historical data leading up to the startup time, preventing the "blank chart" issue on first load.
* **Mock Data Generation**: Simulates currency correlations (independent movements of USD and EUR against JPY) by combining trigonometric functions (`sin`, `cos`) with random noise.
* **Real-Time Clock Management**: Uses `dt_obj.strftime("%H:%M")` as the time key to simplify time-series axis rendering on the frontend.

### B. Frontend: Dynamic Visualization

* **Subplot Configuration**: Utilizes Plotly’s `grid` settings to align USD/JPY and EUR/JPY vertically with independent Y-axes, enabling the simultaneous comparison of currencies with different price ranges.
* **Data Rotation**: Leverages `array.shift()` to ensure only the latest 60 minutes of data are retained, maintaining a fixed-period window while minimizing browser memory load.
* **Layer Design**:
* **Raw Data**: Displayed as a thin black line (`#000000`) representing current prices.
* **MA Lines**: Displayed as thick solid lines in Red, Green, and Blue to represent various trend periods.
* **Vector Plot**: Visualizes the relative strength among three currencies through centroid calculation within an equilateral triangle.



---

## 3. Mathematical Approach

### Centroid Vector Synthesis

For the three vertices $P_{USD}, P_{JPY}, P_{EUR}$, the average coordinates are calculated by multiplying each currency's weight (strength).

$$x = \frac{\sum w_i x_i}{\sum w_i}, \quad y = \frac{\sum w_i y_i}{\sum w_i}$$

This implementation allows for an interface where a user can instantly determine which currencies are being bought or sold based solely on the movement of a single point (or vector).

---

## 4. Future Scalability

* **Live Data Integration**: Replacing the mock generator with modules that fetch real-time rates from OANDA API, yfinance, or other financial data providers.
* **Technical Indicator Integration**: Incorporating calculation logic for Bollinger Bands, RSI, and other advanced technical indicators.
* **Multi-Currency Support**: Expanding vector synthesis to regular polygons (pentagons or hexagons) to include currencies like GBP, AUD, and others.
