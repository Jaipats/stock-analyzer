"""
Stock Analyzer Web Dashboard
Flask application with interactive visualizations.
"""

from flask import Flask, render_template, request, jsonify
import json
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from modules.technical_analysis import TechnicalAnalyzer
from modules.sentiment_analysis import SentimentAnalyzer
from modules.fundamental_analysis import FundamentalAnalyzer
from analyzer import StockAnalyzer

app = Flask(__name__)

# Enable CORS
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    response.headers.add('X-Frame-Options', 'ALLOWALL')
    return response


@app.route('/')
def index():
    """Render main dashboard."""
    return render_template('index.html')


@app.route('/analyze', methods=['POST'])
def analyze():
    """Run analysis on a stock ticker."""
    data = request.get_json()
    ticker = data.get('ticker', 'AAPL').upper()
    
    try:
        analyzer = StockAnalyzer(ticker)
        result = analyzer.analyze()
        
        # Clean up result for JSON serialization
        import numpy as np
        
        def clean_for_json(obj):
            if isinstance(obj, dict):
                return {k: clean_for_json(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [clean_for_json(item) for item in obj]
            elif isinstance(obj, (np.bool_, bool)):
                return bool(obj)
            elif isinstance(obj, (np.integer, np.int64, np.int32)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64, np.float32, float)):
                if obj != obj:  # NaN check
                    return None
                return round(float(obj), 4)
            elif hasattr(obj, 'isoformat'):
                return obj.isoformat()
            elif isinstance(obj, np.ndarray):
                return clean_for_json(obj.tolist())
            else:
                return obj
        
        cleaned_result = clean_for_json(result)
        return jsonify({'success': True, 'data': cleaned_result})
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400


@app.route('/technical/<ticker>')
def technical_only(ticker):
    """Get technical analysis only."""
    try:
        analyzer = TechnicalAnalyzer(ticker.upper())
        result = analyzer.analyze()
        return jsonify({'success': True, 'data': result})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400


@app.route('/sentiment/<ticker>')
def sentiment_only(ticker):
    """Get sentiment analysis only."""
    try:
        analyzer = SentimentAnalyzer(ticker.upper())
        result = analyzer.analyze()
        return jsonify({'success': True, 'data': result})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400


@app.route('/fundamental/<ticker>')
def fundamental_only(ticker):
    """Get fundamental analysis only."""
    try:
        analyzer = FundamentalAnalyzer(ticker.upper())
        result = analyzer.analyze()
        return jsonify({'success': True, 'data': result})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=12000, debug=True)
