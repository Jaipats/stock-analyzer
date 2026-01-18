"""
Technical Analysis Module
Implements various technical indicators and pattern recognition for stock analysis.
Weight: 40% of total score
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import ta
from ta.trend import MACD, EMAIndicator, SMAIndicator, ADXIndicator
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import OnBalanceVolumeIndicator, VolumeWeightedAveragePrice


class TechnicalAnalyzer:
    """Performs technical analysis on stock data."""
    
    def __init__(self, ticker: str, period: str = "1y"):
        self.ticker = ticker.upper()
        self.period = period
        self.data = None
        self.indicators = {}
        self.signals = {}
        self.score = 0
        
    def fetch_data(self) -> pd.DataFrame:
        """Fetch historical stock data from Yahoo Finance."""
        try:
            stock = yf.Ticker(self.ticker)
            self.data = stock.history(period=self.period)
            if self.data.empty:
                raise ValueError(f"No data found for ticker {self.ticker}")
            self.data.index = pd.to_datetime(self.data.index).tz_localize(None)
            return self.data
        except Exception as e:
            raise Exception(f"Error fetching data for {self.ticker}: {str(e)}")
    
    def calculate_moving_averages(self) -> dict:
        """Calculate SMA and EMA for various periods."""
        close = self.data['Close']
        
        # Simple Moving Averages
        self.data['SMA_20'] = SMAIndicator(close, window=20).sma_indicator()
        self.data['SMA_50'] = SMAIndicator(close, window=50).sma_indicator()
        self.data['SMA_200'] = SMAIndicator(close, window=200).sma_indicator()
        
        # Exponential Moving Averages
        self.data['EMA_12'] = EMAIndicator(close, window=12).ema_indicator()
        self.data['EMA_26'] = EMAIndicator(close, window=26).ema_indicator()
        self.data['EMA_50'] = EMAIndicator(close, window=50).ema_indicator()
        
        current_price = close.iloc[-1]
        sma_20 = self.data['SMA_20'].iloc[-1]
        sma_50 = self.data['SMA_50'].iloc[-1]
        sma_200 = self.data['SMA_200'].iloc[-1]
        
        ma_signals = {
            'current_price': current_price,
            'sma_20': sma_20,
            'sma_50': sma_50,
            'sma_200': sma_200,
            'price_vs_sma20': 'above' if current_price > sma_20 else 'below',
            'price_vs_sma50': 'above' if current_price > sma_50 else 'below',
            'price_vs_sma200': 'above' if current_price > sma_200 else 'below',
            'golden_cross': sma_50 > sma_200 if pd.notna(sma_200) else None,
            'death_cross': sma_50 < sma_200 if pd.notna(sma_200) else None,
        }
        
        self.indicators['moving_averages'] = ma_signals
        return ma_signals
    
    def calculate_rsi(self, period: int = 14) -> dict:
        """Calculate Relative Strength Index."""
        rsi = RSIIndicator(self.data['Close'], window=period)
        self.data['RSI'] = rsi.rsi()
        
        current_rsi = self.data['RSI'].iloc[-1]
        
        rsi_signal = {
            'value': current_rsi,
            'period': period,
            'overbought': current_rsi > 70,
            'oversold': current_rsi < 30,
            'neutral': 30 <= current_rsi <= 70,
            'interpretation': 'Overbought' if current_rsi > 70 else ('Oversold' if current_rsi < 30 else 'Neutral')
        }
        
        self.indicators['rsi'] = rsi_signal
        return rsi_signal
    
    def calculate_macd(self) -> dict:
        """Calculate MACD indicator."""
        macd = MACD(self.data['Close'])
        self.data['MACD'] = macd.macd()
        self.data['MACD_Signal'] = macd.macd_signal()
        self.data['MACD_Histogram'] = macd.macd_diff()
        
        current_macd = self.data['MACD'].iloc[-1]
        current_signal = self.data['MACD_Signal'].iloc[-1]
        current_hist = self.data['MACD_Histogram'].iloc[-1]
        prev_hist = self.data['MACD_Histogram'].iloc[-2]
        
        macd_signal = {
            'macd': current_macd,
            'signal': current_signal,
            'histogram': current_hist,
            'bullish_crossover': current_macd > current_signal and self.data['MACD'].iloc[-2] <= self.data['MACD_Signal'].iloc[-2],
            'bearish_crossover': current_macd < current_signal and self.data['MACD'].iloc[-2] >= self.data['MACD_Signal'].iloc[-2],
            'histogram_increasing': current_hist > prev_hist,
            'above_zero': current_macd > 0,
        }
        
        self.indicators['macd'] = macd_signal
        return macd_signal
    
    def calculate_bollinger_bands(self, period: int = 20, std_dev: float = 2.0) -> dict:
        """Calculate Bollinger Bands."""
        bb = BollingerBands(self.data['Close'], window=period, window_dev=std_dev)
        self.data['BB_Upper'] = bb.bollinger_hband()
        self.data['BB_Middle'] = bb.bollinger_mavg()
        self.data['BB_Lower'] = bb.bollinger_lband()
        self.data['BB_Width'] = bb.bollinger_wband()
        self.data['BB_Percent'] = bb.bollinger_pband()
        
        current_price = self.data['Close'].iloc[-1]
        upper = self.data['BB_Upper'].iloc[-1]
        lower = self.data['BB_Lower'].iloc[-1]
        middle = self.data['BB_Middle'].iloc[-1]
        
        bb_signal = {
            'upper': upper,
            'middle': middle,
            'lower': lower,
            'width': self.data['BB_Width'].iloc[-1],
            'percent_b': self.data['BB_Percent'].iloc[-1],
            'near_upper': current_price >= upper * 0.98,
            'near_lower': current_price <= lower * 1.02,
            'squeeze': self.data['BB_Width'].iloc[-1] < self.data['BB_Width'].rolling(20).mean().iloc[-1],
        }
        
        self.indicators['bollinger_bands'] = bb_signal
        return bb_signal
    
    def calculate_stochastic(self, k_period: int = 14, d_period: int = 3) -> dict:
        """Calculate Stochastic Oscillator."""
        stoch = StochasticOscillator(
            self.data['High'], 
            self.data['Low'], 
            self.data['Close'],
            window=k_period,
            smooth_window=d_period
        )
        self.data['Stoch_K'] = stoch.stoch()
        self.data['Stoch_D'] = stoch.stoch_signal()
        
        k = self.data['Stoch_K'].iloc[-1]
        d = self.data['Stoch_D'].iloc[-1]
        
        stoch_signal = {
            'k': k,
            'd': d,
            'overbought': k > 80,
            'oversold': k < 20,
            'bullish_crossover': k > d and self.data['Stoch_K'].iloc[-2] <= self.data['Stoch_D'].iloc[-2],
            'bearish_crossover': k < d and self.data['Stoch_K'].iloc[-2] >= self.data['Stoch_D'].iloc[-2],
        }
        
        self.indicators['stochastic'] = stoch_signal
        return stoch_signal
    
    def calculate_adx(self, period: int = 14) -> dict:
        """Calculate Average Directional Index for trend strength."""
        adx = ADXIndicator(self.data['High'], self.data['Low'], self.data['Close'], window=period)
        self.data['ADX'] = adx.adx()
        self.data['DI_Plus'] = adx.adx_pos()
        self.data['DI_Minus'] = adx.adx_neg()
        
        current_adx = self.data['ADX'].iloc[-1]
        di_plus = self.data['DI_Plus'].iloc[-1]
        di_minus = self.data['DI_Minus'].iloc[-1]
        
        adx_signal = {
            'adx': current_adx,
            'di_plus': di_plus,
            'di_minus': di_minus,
            'strong_trend': current_adx > 25,
            'weak_trend': current_adx < 20,
            'bullish_trend': di_plus > di_minus,
            'bearish_trend': di_minus > di_plus,
        }
        
        self.indicators['adx'] = adx_signal
        return adx_signal
    
    def calculate_atr(self, period: int = 14) -> dict:
        """Calculate Average True Range for volatility."""
        atr = AverageTrueRange(self.data['High'], self.data['Low'], self.data['Close'], window=period)
        self.data['ATR'] = atr.average_true_range()
        
        current_atr = self.data['ATR'].iloc[-1]
        current_price = self.data['Close'].iloc[-1]
        atr_percent = (current_atr / current_price) * 100
        
        atr_signal = {
            'atr': current_atr,
            'atr_percent': atr_percent,
            'high_volatility': atr_percent > 3,
            'low_volatility': atr_percent < 1,
        }
        
        self.indicators['atr'] = atr_signal
        return atr_signal
    
    def calculate_volume_indicators(self) -> dict:
        """Calculate volume-based indicators."""
        obv = OnBalanceVolumeIndicator(self.data['Close'], self.data['Volume'])
        self.data['OBV'] = obv.on_balance_volume()
        
        # Volume moving average
        self.data['Volume_SMA_20'] = self.data['Volume'].rolling(window=20).mean()
        
        current_volume = self.data['Volume'].iloc[-1]
        avg_volume = self.data['Volume_SMA_20'].iloc[-1]
        obv_current = self.data['OBV'].iloc[-1]
        obv_prev = self.data['OBV'].iloc[-5]
        
        volume_signal = {
            'current_volume': current_volume,
            'avg_volume_20': avg_volume,
            'volume_ratio': current_volume / avg_volume if avg_volume > 0 else 1,
            'high_volume': current_volume > avg_volume * 1.5,
            'obv_trend': 'up' if obv_current > obv_prev else 'down',
        }
        
        self.indicators['volume'] = volume_signal
        return volume_signal
    
    def identify_support_resistance(self, lookback: int = 60) -> dict:
        """Identify key support and resistance levels."""
        recent_data = self.data.tail(lookback)
        
        highs = recent_data['High'].values
        lows = recent_data['Low'].values
        
        # Find local maxima and minima
        resistance_levels = []
        support_levels = []
        
        for i in range(2, len(highs) - 2):
            if highs[i] > highs[i-1] and highs[i] > highs[i-2] and highs[i] > highs[i+1] and highs[i] > highs[i+2]:
                resistance_levels.append(highs[i])
            if lows[i] < lows[i-1] and lows[i] < lows[i-2] and lows[i] < lows[i+1] and lows[i] < lows[i+2]:
                support_levels.append(lows[i])
        
        current_price = self.data['Close'].iloc[-1]
        
        # Find nearest support and resistance
        nearest_resistance = min([r for r in resistance_levels if r > current_price], default=None)
        nearest_support = max([s for s in support_levels if s < current_price], default=None)
        
        sr_signal = {
            'resistance_levels': sorted(set(resistance_levels), reverse=True)[:3],
            'support_levels': sorted(set(support_levels))[:3],
            'nearest_resistance': nearest_resistance,
            'nearest_support': nearest_support,
            'distance_to_resistance': ((nearest_resistance - current_price) / current_price * 100) if nearest_resistance else None,
            'distance_to_support': ((current_price - nearest_support) / current_price * 100) if nearest_support else None,
        }
        
        self.indicators['support_resistance'] = sr_signal
        return sr_signal
    
    def calculate_trend(self) -> dict:
        """Determine overall trend direction."""
        close = self.data['Close']
        
        # Short-term trend (20 days)
        short_trend = 'up' if close.iloc[-1] > close.iloc[-20] else 'down'
        short_change = ((close.iloc[-1] - close.iloc[-20]) / close.iloc[-20]) * 100
        
        # Medium-term trend (50 days)
        medium_trend = 'up' if close.iloc[-1] > close.iloc[-50] else 'down'
        medium_change = ((close.iloc[-1] - close.iloc[-50]) / close.iloc[-50]) * 100
        
        # Long-term trend (200 days if available)
        if len(close) >= 200:
            long_trend = 'up' if close.iloc[-1] > close.iloc[-200] else 'down'
            long_change = ((close.iloc[-1] - close.iloc[-200]) / close.iloc[-200]) * 100
        else:
            long_trend = None
            long_change = None
        
        trend_signal = {
            'short_term': {'direction': short_trend, 'change_percent': short_change},
            'medium_term': {'direction': medium_trend, 'change_percent': medium_change},
            'long_term': {'direction': long_trend, 'change_percent': long_change},
            'overall': self._determine_overall_trend(short_trend, medium_trend, long_trend),
        }
        
        self.indicators['trend'] = trend_signal
        return trend_signal
    
    def _determine_overall_trend(self, short: str, medium: str, long: str) -> str:
        """Determine overall trend from multiple timeframes."""
        trends = [short, medium]
        if long:
            trends.append(long)
        
        up_count = trends.count('up')
        if up_count == len(trends):
            return 'strong_bullish'
        elif up_count >= len(trends) / 2:
            return 'bullish'
        elif up_count == 0:
            return 'strong_bearish'
        else:
            return 'bearish'
    
    def calculate_all_indicators(self) -> dict:
        """Calculate all technical indicators."""
        if self.data is None:
            self.fetch_data()
        
        self.calculate_moving_averages()
        self.calculate_rsi()
        self.calculate_macd()
        self.calculate_bollinger_bands()
        self.calculate_stochastic()
        self.calculate_adx()
        self.calculate_atr()
        self.calculate_volume_indicators()
        self.identify_support_resistance()
        self.calculate_trend()
        
        return self.indicators
    
    def generate_signals(self) -> dict:
        """Generate trading signals based on indicators."""
        if not self.indicators:
            self.calculate_all_indicators()
        
        signals = {
            'bullish': [],
            'bearish': [],
            'neutral': []
        }
        
        # Moving Average signals
        ma = self.indicators.get('moving_averages', {})
        if ma.get('price_vs_sma20') == 'above' and ma.get('price_vs_sma50') == 'above':
            signals['bullish'].append('Price above SMA20 and SMA50')
        elif ma.get('price_vs_sma20') == 'below' and ma.get('price_vs_sma50') == 'below':
            signals['bearish'].append('Price below SMA20 and SMA50')
        
        if ma.get('golden_cross'):
            signals['bullish'].append('Golden Cross (SMA50 > SMA200)')
        elif ma.get('death_cross'):
            signals['bearish'].append('Death Cross (SMA50 < SMA200)')
        
        # RSI signals
        rsi = self.indicators.get('rsi', {})
        if rsi.get('oversold'):
            signals['bullish'].append(f"RSI oversold ({rsi.get('value', 0):.1f})")
        elif rsi.get('overbought'):
            signals['bearish'].append(f"RSI overbought ({rsi.get('value', 0):.1f})")
        else:
            signals['neutral'].append(f"RSI neutral ({rsi.get('value', 0):.1f})")
        
        # MACD signals
        macd = self.indicators.get('macd', {})
        if macd.get('bullish_crossover'):
            signals['bullish'].append('MACD bullish crossover')
        elif macd.get('bearish_crossover'):
            signals['bearish'].append('MACD bearish crossover')
        
        if macd.get('above_zero') and macd.get('histogram_increasing'):
            signals['bullish'].append('MACD above zero with increasing momentum')
        elif not macd.get('above_zero') and not macd.get('histogram_increasing'):
            signals['bearish'].append('MACD below zero with decreasing momentum')
        
        # Bollinger Bands signals
        bb = self.indicators.get('bollinger_bands', {})
        if bb.get('near_lower'):
            signals['bullish'].append('Price near lower Bollinger Band')
        elif bb.get('near_upper'):
            signals['bearish'].append('Price near upper Bollinger Band')
        
        if bb.get('squeeze'):
            signals['neutral'].append('Bollinger Band squeeze (potential breakout)')
        
        # Stochastic signals
        stoch = self.indicators.get('stochastic', {})
        if stoch.get('oversold') and stoch.get('bullish_crossover'):
            signals['bullish'].append('Stochastic bullish crossover from oversold')
        elif stoch.get('overbought') and stoch.get('bearish_crossover'):
            signals['bearish'].append('Stochastic bearish crossover from overbought')
        
        # ADX signals
        adx = self.indicators.get('adx', {})
        if adx.get('strong_trend') and adx.get('bullish_trend'):
            signals['bullish'].append('Strong bullish trend (ADX)')
        elif adx.get('strong_trend') and adx.get('bearish_trend'):
            signals['bearish'].append('Strong bearish trend (ADX)')
        
        # Volume signals
        vol = self.indicators.get('volume', {})
        if vol.get('high_volume') and vol.get('obv_trend') == 'up':
            signals['bullish'].append('High volume with positive OBV trend')
        elif vol.get('high_volume') and vol.get('obv_trend') == 'down':
            signals['bearish'].append('High volume with negative OBV trend')
        
        # Trend signals
        trend = self.indicators.get('trend', {})
        if trend.get('overall') == 'strong_bullish':
            signals['bullish'].append('Strong bullish trend across timeframes')
        elif trend.get('overall') == 'strong_bearish':
            signals['bearish'].append('Strong bearish trend across timeframes')
        
        self.signals = signals
        return signals
    
    def calculate_score(self) -> dict:
        """Calculate technical analysis score (-1 to +1)."""
        if not self.signals:
            self.generate_signals()
        
        bullish_count = len(self.signals['bullish'])
        bearish_count = len(self.signals['bearish'])
        total_signals = bullish_count + bearish_count
        
        if total_signals == 0:
            self.score = 0
        else:
            self.score = (bullish_count - bearish_count) / total_signals
        
        # Determine recommendation
        if self.score > 0.3:
            recommendation = 'BULLISH'
        elif self.score < -0.3:
            recommendation = 'BEARISH'
        else:
            recommendation = 'NEUTRAL'
        
        return {
            'score': self.score,
            'recommendation': recommendation,
            'bullish_signals': bullish_count,
            'bearish_signals': bearish_count,
            'neutral_signals': len(self.signals['neutral']),
            'signals': self.signals,
        }
    
    def get_chart_data(self) -> dict:
        """Get data formatted for charting."""
        if self.data is None:
            self.fetch_data()
            self.calculate_all_indicators()
        
        # Get last 90 days for charting
        chart_data = self.data.tail(90).copy()
        chart_data.index = chart_data.index.strftime('%Y-%m-%d')
        
        return {
            'dates': chart_data.index.tolist(),
            'ohlc': {
                'open': chart_data['Open'].tolist(),
                'high': chart_data['High'].tolist(),
                'low': chart_data['Low'].tolist(),
                'close': chart_data['Close'].tolist(),
            },
            'volume': chart_data['Volume'].tolist(),
            'sma_20': chart_data['SMA_20'].tolist() if 'SMA_20' in chart_data else [],
            'sma_50': chart_data['SMA_50'].tolist() if 'SMA_50' in chart_data else [],
            'rsi': chart_data['RSI'].tolist() if 'RSI' in chart_data else [],
            'macd': chart_data['MACD'].tolist() if 'MACD' in chart_data else [],
            'macd_signal': chart_data['MACD_Signal'].tolist() if 'MACD_Signal' in chart_data else [],
            'macd_histogram': chart_data['MACD_Histogram'].tolist() if 'MACD_Histogram' in chart_data else [],
            'bb_upper': chart_data['BB_Upper'].tolist() if 'BB_Upper' in chart_data else [],
            'bb_lower': chart_data['BB_Lower'].tolist() if 'BB_Lower' in chart_data else [],
        }
    
    def analyze(self) -> dict:
        """Run complete technical analysis."""
        self.fetch_data()
        self.calculate_all_indicators()
        self.generate_signals()
        score_data = self.calculate_score()
        
        return {
            'ticker': self.ticker,
            'analysis_type': 'technical',
            'weight': 0.4,
            'indicators': self.indicators,
            'score_data': score_data,
            'chart_data': self.get_chart_data(),
        }


if __name__ == "__main__":
    # Test the module
    analyzer = TechnicalAnalyzer("AAPL")
    result = analyzer.analyze()
    print(f"Technical Analysis for {result['ticker']}")
    print(f"Score: {result['score_data']['score']:.2f}")
    print(f"Recommendation: {result['score_data']['recommendation']}")
    print(f"\nBullish Signals: {result['score_data']['signals']['bullish']}")
    print(f"Bearish Signals: {result['score_data']['signals']['bearish']}")
