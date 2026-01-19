# 📈 Stock Analyzer

A comprehensive stock analysis tool that combines **Technical Analysis**, **Sentiment Analysis**, and **Fundamental Analysis** to provide weighted composite scores and trading signals.

## Features

### Technical Analysis (40% weight)
- Moving Averages (SMA 20, 50, 200 & EMA)
- RSI (Relative Strength Index)
- MACD (Moving Average Convergence Divergence)
- Bollinger Bands
- Stochastic Oscillator
- ADX (Average Directional Index)
- ATR (Average True Range)
- Volume Analysis (OBV)
- Support/Resistance Levels
- Trend Analysis

### Sentiment Analysis (40% weight)
- News sentiment from Yahoo Finance, Finviz, Google News
- Social media sentiment from StockTwits
- Insider trading activity (SEC Form 4)
- Analyst ratings and recommendations
- Short interest data

### Fundamental Analysis (20% weight)
- Valuation metrics (P/E, PEG, P/B, P/S, EV/EBITDA)
- Growth metrics (Revenue, Earnings growth)
- Profitability (Margins, ROE, ROA)
- Financial health (Debt/Equity, Current Ratio, FCF)
- Dividend analysis

## Installation

```bash
pip install -r requirements.txt
```

> **Note:** Depending on your system configuration, you may need to use `python3` and `pip3` instead of `python` and `pip`.

### Local Development Setup (Recommended)

For local development, it's recommended to use a virtual environment to avoid conflicts with system packages:

#### On macOS/Linux:

```bash
# Create a virtual environment
python3 -m venv venv

# Activate the virtual environment
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# When done, deactivate the virtual environment
deactivate
```

#### On Windows:

```bash
# Create a virtual environment
python -m venv venv

# Activate the virtual environment
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# When done, deactivate the virtual environment
deactivate
```

## Usage

### Command Line
```bash
cd stock_analyzer
python analyzer.py AAPL
```

### Web Dashboard
```bash
cd stock_analyzer
python app.py
```
Then open http://localhost:12000 in your browser.

### Python API
```python
from stock_analyzer.analyzer import StockAnalyzer

analyzer = StockAnalyzer("AAPL")
result = analyzer.analyze()
analyzer.print_summary(result)
```

## Project Structure

```
stock_analyzer/
├── modules/
│   ├── technical_analysis.py   # Technical indicators & signals
│   ├── sentiment_analysis.py   # News, social, insider sentiment
│   └── fundamental_analysis.py # Financial metrics & valuation
├── templates/
│   └── index.html              # Web dashboard template
├── analyzer.py                 # Main analyzer with composite scoring
└── app.py                      # Flask web application
```

## Free Data Sources Used

| Source | URL | Data Provided |
|--------|-----|---------------|
| **Yahoo Finance** | [finance.yahoo.com](https://finance.yahoo.com) | Historical prices, financials, insider data |
| **Finviz** | [finviz.com](https://finviz.com) | News, stock screener data |
| **Google News RSS** | [news.google.com](https://news.google.com) | News headlines |
| **StockTwits API** | [stocktwits.com](https://stocktwits.com) | Social sentiment |
| **SEC EDGAR** | [sec.gov/edgar](https://www.sec.gov/edgar) | Insider transactions (via yfinance) |

## Scoring System

The composite score ranges from -1 (Strong Sell) to +1 (Strong Buy):

| Score Range | Recommendation |
|-------------|----------------|
| > 0.25      | STRONG BUY     |
| 0.1 to 0.25 | BUY            |
| -0.1 to 0.1 | HOLD           |
| -0.25 to -0.1| SELL          |
| < -0.25     | STRONG SELL    |

## Requirements

- Python 3.8+
- yfinance
- pandas
- numpy
- ta (Technical Analysis library)
- vaderSentiment
- flask
- plotly
- beautifulsoup4
- feedparser
- requests

## Disclaimer

⚠️ **This application is for informational and educational purposes only. It does not constitute financial advice. Always do your own research and consult with a qualified financial advisor before making investment decisions.**

## License

MIT License
