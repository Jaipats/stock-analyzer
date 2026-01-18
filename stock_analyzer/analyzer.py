"""
Stock Analyzer - Main Module
Combines Technical, Sentiment, and Fundamental analysis with weighted scoring.
"""

from modules.technical_analysis import TechnicalAnalyzer
from modules.sentiment_analysis import SentimentAnalyzer
from modules.fundamental_analysis import FundamentalAnalyzer
from datetime import datetime
import json


class StockAnalyzer:
    """
    Comprehensive stock analyzer combining multiple analysis types.
    
    Weights:
    - Technical Analysis: 40%
    - Sentiment Analysis: 40%
    - Fundamental Analysis: 20%
    """
    
    def __init__(self, ticker: str, weights: dict = None):
        self.ticker = ticker.upper()
        self.weights = weights or {
            'technical': 0.40,
            'sentiment': 0.40,
            'fundamental': 0.20
        }
        
        self.technical_result = None
        self.sentiment_result = None
        self.fundamental_result = None
        self.composite_score = 0
        self.recommendation = 'NEUTRAL'
        
    def run_technical_analysis(self) -> dict:
        """Run technical analysis."""
        print(f"Running Technical Analysis for {self.ticker}...")
        analyzer = TechnicalAnalyzer(self.ticker)
        self.technical_result = analyzer.analyze()
        return self.technical_result
    
    def run_sentiment_analysis(self) -> dict:
        """Run sentiment analysis."""
        print(f"Running Sentiment Analysis for {self.ticker}...")
        analyzer = SentimentAnalyzer(self.ticker)
        self.sentiment_result = analyzer.analyze()
        return self.sentiment_result
    
    def run_fundamental_analysis(self) -> dict:
        """Run fundamental analysis."""
        print(f"Running Fundamental Analysis for {self.ticker}...")
        analyzer = FundamentalAnalyzer(self.ticker)
        self.fundamental_result = analyzer.analyze()
        return self.fundamental_result
    
    def calculate_composite_score(self) -> dict:
        """Calculate weighted composite score from all analyses."""
        scores = {}
        weighted_sum = 0
        total_weight = 0
        
        if self.technical_result:
            tech_score = self.technical_result['score_data']['score']
            scores['technical'] = {
                'score': tech_score,
                'weight': self.weights['technical'],
                'weighted_score': tech_score * self.weights['technical'],
                'recommendation': self.technical_result['score_data']['recommendation']
            }
            weighted_sum += tech_score * self.weights['technical']
            total_weight += self.weights['technical']
        
        if self.sentiment_result:
            sent_score = self.sentiment_result['score_data']['score']
            scores['sentiment'] = {
                'score': sent_score,
                'weight': self.weights['sentiment'],
                'weighted_score': sent_score * self.weights['sentiment'],
                'recommendation': self.sentiment_result['score_data']['recommendation']
            }
            weighted_sum += sent_score * self.weights['sentiment']
            total_weight += self.weights['sentiment']
        
        if self.fundamental_result:
            fund_score = self.fundamental_result['score_data']['score']
            scores['fundamental'] = {
                'score': fund_score,
                'weight': self.weights['fundamental'],
                'weighted_score': fund_score * self.weights['fundamental'],
                'recommendation': self.fundamental_result['score_data']['recommendation']
            }
            weighted_sum += fund_score * self.weights['fundamental']
            total_weight += self.weights['fundamental']
        
        # Calculate composite score
        self.composite_score = weighted_sum / total_weight if total_weight > 0 else 0
        
        # Determine overall recommendation
        if self.composite_score > 0.25:
            self.recommendation = 'STRONG BUY'
        elif self.composite_score > 0.1:
            self.recommendation = 'BUY'
        elif self.composite_score > -0.1:
            self.recommendation = 'HOLD'
        elif self.composite_score > -0.25:
            self.recommendation = 'SELL'
        else:
            self.recommendation = 'STRONG SELL'
        
        # Calculate confidence level based on signal agreement
        recommendations = [s['recommendation'] for s in scores.values()]
        bullish_count = sum(1 for r in recommendations if r == 'BULLISH')
        bearish_count = sum(1 for r in recommendations if r == 'BEARISH')
        
        if bullish_count == 3 or bearish_count == 3:
            confidence = 'HIGH'
        elif bullish_count >= 2 or bearish_count >= 2:
            confidence = 'MEDIUM'
        else:
            confidence = 'LOW'
        
        return {
            'composite_score': self.composite_score,
            'recommendation': self.recommendation,
            'confidence': confidence,
            'component_scores': scores,
        }
    
    def get_all_signals(self) -> dict:
        """Aggregate all signals from all analyses."""
        all_bullish = []
        all_bearish = []
        all_neutral = []
        
        if self.technical_result:
            signals = self.technical_result['score_data']['signals']
            all_bullish.extend([f"[Technical] {s}" for s in signals.get('bullish', [])])
            all_bearish.extend([f"[Technical] {s}" for s in signals.get('bearish', [])])
            all_neutral.extend([f"[Technical] {s}" for s in signals.get('neutral', [])])
        
        if self.sentiment_result:
            signals = self.sentiment_result['score_data']['signals']
            all_bullish.extend([f"[Sentiment] {s}" for s in signals.get('bullish', [])])
            all_bearish.extend([f"[Sentiment] {s}" for s in signals.get('bearish', [])])
            all_neutral.extend([f"[Sentiment] {s}" for s in signals.get('neutral', [])])
        
        if self.fundamental_result:
            signals = self.fundamental_result['score_data']['signals']
            all_bullish.extend([f"[Fundamental] {s}" for s in signals.get('bullish', [])])
            all_bearish.extend([f"[Fundamental] {s}" for s in signals.get('bearish', [])])
            all_neutral.extend([f"[Fundamental] {s}" for s in signals.get('neutral', [])])
        
        return {
            'bullish': all_bullish,
            'bearish': all_bearish,
            'neutral': all_neutral,
            'total_bullish': len(all_bullish),
            'total_bearish': len(all_bearish),
            'total_neutral': len(all_neutral),
        }
    
    def analyze(self) -> dict:
        """Run complete analysis."""
        print(f"\n{'='*60}")
        print(f"COMPREHENSIVE STOCK ANALYSIS: {self.ticker}")
        print(f"{'='*60}\n")
        
        # Run all analyses
        self.run_technical_analysis()
        self.run_sentiment_analysis()
        self.run_fundamental_analysis()
        
        # Calculate composite score
        composite = self.calculate_composite_score()
        
        # Get all signals
        all_signals = self.get_all_signals()
        
        # Build complete result
        result = {
            'ticker': self.ticker,
            'analysis_date': datetime.now().isoformat(),
            'composite': composite,
            'all_signals': all_signals,
            'technical': self.technical_result,
            'sentiment': self.sentiment_result,
            'fundamental': self.fundamental_result,
        }
        
        return result
    
    def print_summary(self, result: dict = None):
        """Print a formatted summary of the analysis."""
        if result is None:
            result = self.analyze()
        
        print(f"\n{'='*60}")
        print(f"ANALYSIS SUMMARY: {self.ticker}")
        print(f"{'='*60}")
        
        composite = result['composite']
        print(f"\n📊 COMPOSITE SCORE: {composite['composite_score']:.3f}")
        print(f"📈 RECOMMENDATION: {composite['recommendation']}")
        print(f"🎯 CONFIDENCE: {composite['confidence']}")
        
        print(f"\n{'─'*40}")
        print("COMPONENT SCORES:")
        print(f"{'─'*40}")
        
        for name, data in composite['component_scores'].items():
            print(f"  {name.upper():15} Score: {data['score']:+.3f} (Weight: {data['weight']:.0%}) → {data['recommendation']}")
        
        print(f"\n{'─'*40}")
        print("SIGNALS SUMMARY:")
        print(f"{'─'*40}")
        
        signals = result['all_signals']
        print(f"  ✅ Bullish Signals: {signals['total_bullish']}")
        print(f"  ❌ Bearish Signals: {signals['total_bearish']}")
        print(f"  ⚪ Neutral Signals: {signals['total_neutral']}")
        
        if signals['bullish']:
            print(f"\n  TOP BULLISH SIGNALS:")
            for s in signals['bullish'][:5]:
                print(f"    • {s}")
        
        if signals['bearish']:
            print(f"\n  TOP BEARISH SIGNALS:")
            for s in signals['bearish'][:5]:
                print(f"    • {s}")
        
        # Key metrics
        if result.get('fundamental'):
            fund = result['fundamental']
            print(f"\n{'─'*40}")
            print("KEY METRICS:")
            print(f"{'─'*40}")
            
            company = fund.get('company_info', {})
            print(f"  Company: {company.get('name', 'N/A')}")
            print(f"  Sector: {company.get('sector', 'N/A')}")
            
            val = fund.get('valuation', {})
            print(f"  P/E Ratio: {val.get('pe_ratio', 'N/A')}")
            print(f"  PEG Ratio: {val.get('peg_ratio', 'N/A')}")
            
            growth = fund.get('growth', {})
            rev_growth = growth.get('revenue_growth_yoy')
            print(f"  Revenue Growth: {rev_growth:.1f}%" if rev_growth else "  Revenue Growth: N/A")
            
            health = fund.get('financial_health', {})
            print(f"  Debt/Equity: {health.get('debt_to_equity', 'N/A')}")
        
        print(f"\n{'='*60}\n")
        
        return result


def analyze_stock(ticker: str, weights: dict = None) -> dict:
    """Convenience function to analyze a stock."""
    analyzer = StockAnalyzer(ticker, weights)
    return analyzer.analyze()


if __name__ == "__main__":
    import sys
    
    ticker = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
    
    analyzer = StockAnalyzer(ticker)
    result = analyzer.analyze()
    analyzer.print_summary(result)
