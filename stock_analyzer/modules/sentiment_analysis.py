"""
Sentiment Analysis Module
Analyzes market sentiment from news, social media, and insider activity.
Weight: 40% of total score
"""

import requests
from bs4 import BeautifulSoup
import feedparser
from datetime import datetime, timedelta
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import yfinance as yf
import re
from typing import List, Dict, Optional
import time


class SentimentAnalyzer:
    """Performs sentiment analysis on stock-related content."""
    
    def __init__(self, ticker: str):
        self.ticker = ticker.upper()
        self.vader = SentimentIntensityAnalyzer()
        self.news_sentiment = {}
        self.social_sentiment = {}
        self.insider_data = {}
        self.analyst_data = {}
        self.signals = {}
        self.score = 0
        
        # Get company info
        try:
            stock = yf.Ticker(self.ticker)
            self.company_name = stock.info.get('shortName', self.ticker)
            self.company_info = stock.info
        except:
            self.company_name = self.ticker
            self.company_info = {}
    
    def _analyze_text_sentiment(self, text: str) -> dict:
        """Analyze sentiment of text using VADER."""
        scores = self.vader.polarity_scores(text)
        return {
            'compound': scores['compound'],
            'positive': scores['pos'],
            'negative': scores['neg'],
            'neutral': scores['neu'],
            'label': 'positive' if scores['compound'] > 0.05 else ('negative' if scores['compound'] < -0.05 else 'neutral')
        }
    
    def fetch_yahoo_news(self) -> List[Dict]:
        """Fetch news from Yahoo Finance RSS feed."""
        news_items = []
        
        try:
            # Yahoo Finance RSS feed for the ticker
            url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={self.ticker}&region=US&lang=en-US"
            feed = feedparser.parse(url)
            
            for entry in feed.entries[:15]:
                title = entry.get('title', '')
                summary = entry.get('summary', '')
                published = entry.get('published', '')
                link = entry.get('link', '')
                
                # Analyze sentiment
                text = f"{title} {summary}"
                sentiment = self._analyze_text_sentiment(text)
                
                news_items.append({
                    'title': title,
                    'summary': summary[:200] + '...' if len(summary) > 200 else summary,
                    'published': published,
                    'link': link,
                    'source': 'Yahoo Finance',
                    'sentiment': sentiment
                })
        except Exception as e:
            print(f"Error fetching Yahoo news: {e}")
        
        return news_items
    
    def fetch_finviz_news(self) -> List[Dict]:
        """Fetch news from Finviz."""
        news_items = []
        
        try:
            url = f"https://finviz.com/quote.ashx?t={self.ticker}"
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            response = requests.get(url, headers=headers, timeout=10)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Find news table
            news_table = soup.find('table', {'id': 'news-table'})
            if news_table:
                rows = news_table.find_all('tr')[:10]
                
                for row in rows:
                    link_tag = row.find('a', {'class': 'tab-link-news'})
                    if link_tag:
                        title = link_tag.text.strip()
                        link = link_tag.get('href', '')
                        
                        # Get date/time
                        td = row.find('td')
                        date_text = td.text.strip() if td else ''
                        
                        sentiment = self._analyze_text_sentiment(title)
                        
                        news_items.append({
                            'title': title,
                            'summary': '',
                            'published': date_text,
                            'link': link,
                            'source': 'Finviz',
                            'sentiment': sentiment
                        })
        except Exception as e:
            print(f"Error fetching Finviz news: {e}")
        
        return news_items
    
    def fetch_google_news(self) -> List[Dict]:
        """Fetch news from Google News RSS."""
        news_items = []
        
        try:
            # Search for company news (URL encode the query)
            import urllib.parse
            search_query = urllib.parse.quote(f"{self.ticker} stock")
            url = f"https://news.google.com/rss/search?q={search_query}&hl=en-US&gl=US&ceid=US:en"
            feed = feedparser.parse(url)
            
            for entry in feed.entries[:10]:
                title = entry.get('title', '')
                published = entry.get('published', '')
                link = entry.get('link', '')
                
                sentiment = self._analyze_text_sentiment(title)
                
                news_items.append({
                    'title': title,
                    'summary': '',
                    'published': published,
                    'link': link,
                    'source': 'Google News',
                    'sentiment': sentiment
                })
        except Exception as e:
            print(f"Error fetching Google news: {e}")
        
        return news_items
    
    def analyze_news_sentiment(self) -> dict:
        """Aggregate and analyze news sentiment."""
        all_news = []
        
        # Fetch from multiple sources
        all_news.extend(self.fetch_yahoo_news())
        time.sleep(0.5)  # Rate limiting
        all_news.extend(self.fetch_finviz_news())
        time.sleep(0.5)
        all_news.extend(self.fetch_google_news())
        
        if not all_news:
            self.news_sentiment = {
                'articles': [],
                'total_count': 0,
                'avg_sentiment': 0,
                'positive_count': 0,
                'negative_count': 0,
                'neutral_count': 0,
                'sentiment_label': 'neutral'
            }
            return self.news_sentiment
        
        # Calculate aggregate sentiment
        sentiments = [article['sentiment']['compound'] for article in all_news]
        avg_sentiment = sum(sentiments) / len(sentiments)
        
        positive_count = sum(1 for s in sentiments if s > 0.05)
        negative_count = sum(1 for s in sentiments if s < -0.05)
        neutral_count = len(sentiments) - positive_count - negative_count
        
        self.news_sentiment = {
            'articles': all_news[:20],  # Keep top 20
            'total_count': len(all_news),
            'avg_sentiment': avg_sentiment,
            'positive_count': positive_count,
            'negative_count': negative_count,
            'neutral_count': neutral_count,
            'sentiment_label': 'positive' if avg_sentiment > 0.1 else ('negative' if avg_sentiment < -0.1 else 'neutral')
        }
        
        return self.news_sentiment
    
    def fetch_stocktwits_sentiment(self) -> dict:
        """Fetch sentiment from StockTwits API."""
        try:
            url = f"https://api.stocktwits.com/api/2/streams/symbol/{self.ticker}.json"
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                messages = data.get('messages', [])
                
                bullish = 0
                bearish = 0
                total = len(messages)
                
                recent_messages = []
                for msg in messages[:20]:
                    sentiment = msg.get('entities', {}).get('sentiment', {})
                    if sentiment:
                        if sentiment.get('basic') == 'Bullish':
                            bullish += 1
                        elif sentiment.get('basic') == 'Bearish':
                            bearish += 1
                    
                    recent_messages.append({
                        'body': msg.get('body', '')[:200],
                        'created_at': msg.get('created_at', ''),
                        'sentiment': sentiment.get('basic', 'Unknown'),
                        'user': msg.get('user', {}).get('username', 'Unknown')
                    })
                
                return {
                    'messages': recent_messages,
                    'total_messages': total,
                    'bullish_count': bullish,
                    'bearish_count': bearish,
                    'bullish_ratio': bullish / total if total > 0 else 0.5,
                    'sentiment_score': (bullish - bearish) / total if total > 0 else 0
                }
        except Exception as e:
            print(f"Error fetching StockTwits: {e}")
        
        return {
            'messages': [],
            'total_messages': 0,
            'bullish_count': 0,
            'bearish_count': 0,
            'bullish_ratio': 0.5,
            'sentiment_score': 0
        }
    
    def analyze_social_sentiment(self) -> dict:
        """Analyze social media sentiment."""
        stocktwits = self.fetch_stocktwits_sentiment()
        
        self.social_sentiment = {
            'stocktwits': stocktwits,
            'overall_score': stocktwits['sentiment_score'],
            'sentiment_label': 'bullish' if stocktwits['sentiment_score'] > 0.1 else (
                'bearish' if stocktwits['sentiment_score'] < -0.1 else 'neutral'
            )
        }
        
        return self.social_sentiment
    
    def fetch_insider_transactions(self) -> dict:
        """Fetch insider trading data from Yahoo Finance."""
        try:
            stock = yf.Ticker(self.ticker)
            
            # Get insider transactions
            insider_transactions = stock.insider_transactions
            
            if insider_transactions is not None and not insider_transactions.empty:
                recent = insider_transactions.head(20)
                
                buys = 0
                sells = 0
                buy_value = 0
                sell_value = 0
                
                transactions = []
                for _, row in recent.iterrows():
                    trans_type = str(row.get('Text', '')).lower()
                    shares = row.get('Shares', 0)
                    value = row.get('Value', 0)
                    
                    if 'buy' in trans_type or 'purchase' in trans_type:
                        buys += 1
                        buy_value += value if value else 0
                    elif 'sell' in trans_type or 'sale' in trans_type:
                        sells += 1
                        sell_value += value if value else 0
                    
                    transactions.append({
                        'insider': row.get('Insider', 'Unknown'),
                        'relation': row.get('Relationship', 'Unknown'),
                        'transaction': row.get('Text', 'Unknown'),
                        'shares': shares,
                        'value': value,
                        'date': str(row.get('Start Date', ''))
                    })
                
                return {
                    'transactions': transactions,
                    'buy_count': buys,
                    'sell_count': sells,
                    'buy_value': buy_value,
                    'sell_value': sell_value,
                    'net_sentiment': 'bullish' if buys > sells else ('bearish' if sells > buys else 'neutral'),
                    'insider_score': (buys - sells) / (buys + sells) if (buys + sells) > 0 else 0
                }
        except Exception as e:
            print(f"Error fetching insider data: {e}")
        
        return {
            'transactions': [],
            'buy_count': 0,
            'sell_count': 0,
            'buy_value': 0,
            'sell_value': 0,
            'net_sentiment': 'neutral',
            'insider_score': 0
        }
    
    def fetch_analyst_ratings(self) -> dict:
        """Fetch analyst ratings and recommendations."""
        try:
            stock = yf.Ticker(self.ticker)
            
            # Get recommendations
            recommendations = stock.recommendations
            
            if recommendations is not None and not recommendations.empty:
                recent = recommendations.tail(20)
                
                strong_buy = 0
                buy = 0
                hold = 0
                sell = 0
                strong_sell = 0
                
                for _, row in recent.iterrows():
                    grade = str(row.get('To Grade', '')).lower()
                    
                    if 'strong buy' in grade:
                        strong_buy += 1
                    elif 'buy' in grade or 'outperform' in grade or 'overweight' in grade:
                        buy += 1
                    elif 'hold' in grade or 'neutral' in grade or 'equal' in grade:
                        hold += 1
                    elif 'strong sell' in grade:
                        strong_sell += 1
                    elif 'sell' in grade or 'underperform' in grade or 'underweight' in grade:
                        sell += 1
                
                total = strong_buy + buy + hold + sell + strong_sell
                if total > 0:
                    # Weighted score: strong buy=2, buy=1, hold=0, sell=-1, strong sell=-2
                    score = (strong_buy * 2 + buy * 1 + hold * 0 + sell * -1 + strong_sell * -2) / total
                    normalized_score = score / 2  # Normalize to -1 to 1
                else:
                    normalized_score = 0
                
                return {
                    'strong_buy': strong_buy,
                    'buy': buy,
                    'hold': hold,
                    'sell': sell,
                    'strong_sell': strong_sell,
                    'total_ratings': total,
                    'analyst_score': normalized_score,
                    'consensus': 'buy' if normalized_score > 0.2 else ('sell' if normalized_score < -0.2 else 'hold')
                }
        except Exception as e:
            print(f"Error fetching analyst ratings: {e}")
        
        return {
            'strong_buy': 0,
            'buy': 0,
            'hold': 0,
            'sell': 0,
            'strong_sell': 0,
            'total_ratings': 0,
            'analyst_score': 0,
            'consensus': 'neutral'
        }
    
    def fetch_short_interest(self) -> dict:
        """Fetch short interest data."""
        try:
            stock = yf.Ticker(self.ticker)
            info = stock.info
            
            short_ratio = info.get('shortRatio', 0)
            short_percent = info.get('shortPercentOfFloat', 0)
            
            # High short interest can be bearish (or bullish for squeeze potential)
            if short_percent and short_percent > 0.2:  # > 20%
                sentiment = 'high_short_interest'
            elif short_percent and short_percent > 0.1:  # > 10%
                sentiment = 'moderate_short_interest'
            else:
                sentiment = 'low_short_interest'
            
            return {
                'short_ratio': short_ratio,
                'short_percent_of_float': short_percent,
                'sentiment': sentiment,
                'short_score': -short_percent if short_percent else 0  # Higher short = more bearish
            }
        except Exception as e:
            print(f"Error fetching short interest: {e}")
        
        return {
            'short_ratio': 0,
            'short_percent_of_float': 0,
            'sentiment': 'unknown',
            'short_score': 0
        }
    
    def generate_signals(self) -> dict:
        """Generate sentiment signals."""
        signals = {
            'bullish': [],
            'bearish': [],
            'neutral': []
        }
        
        # News sentiment signals
        if self.news_sentiment:
            avg_sent = self.news_sentiment.get('avg_sentiment', 0)
            if avg_sent > 0.15:
                signals['bullish'].append(f"Positive news sentiment ({avg_sent:.2f})")
            elif avg_sent < -0.15:
                signals['bearish'].append(f"Negative news sentiment ({avg_sent:.2f})")
            else:
                signals['neutral'].append(f"Neutral news sentiment ({avg_sent:.2f})")
            
            pos_ratio = self.news_sentiment.get('positive_count', 0) / max(self.news_sentiment.get('total_count', 1), 1)
            if pos_ratio > 0.6:
                signals['bullish'].append(f"High positive news ratio ({pos_ratio:.0%})")
            elif pos_ratio < 0.3:
                signals['bearish'].append(f"Low positive news ratio ({pos_ratio:.0%})")
        
        # Social sentiment signals
        if self.social_sentiment:
            social_score = self.social_sentiment.get('overall_score', 0)
            if social_score > 0.2:
                signals['bullish'].append(f"Bullish social sentiment ({social_score:.2f})")
            elif social_score < -0.2:
                signals['bearish'].append(f"Bearish social sentiment ({social_score:.2f})")
        
        # Insider signals
        if self.insider_data:
            insider_score = self.insider_data.get('insider_score', 0)
            if insider_score > 0.3:
                signals['bullish'].append("Net insider buying")
            elif insider_score < -0.3:
                signals['bearish'].append("Net insider selling")
        
        # Analyst signals
        if self.analyst_data:
            analyst_score = self.analyst_data.get('analyst_score', 0)
            if analyst_score > 0.3:
                signals['bullish'].append(f"Analyst consensus: Buy ({self.analyst_data.get('consensus', '')})")
            elif analyst_score < -0.3:
                signals['bearish'].append(f"Analyst consensus: Sell ({self.analyst_data.get('consensus', '')})")
        
        # Short interest signals
        short_data = self.fetch_short_interest()
        if short_data.get('short_percent_of_float', 0) > 0.15:
            signals['bearish'].append(f"High short interest ({short_data['short_percent_of_float']:.1%})")
        
        self.signals = signals
        return signals
    
    def calculate_score(self) -> dict:
        """Calculate sentiment analysis score (-1 to +1)."""
        scores = []
        weights = []
        
        # News sentiment (weight: 35%)
        if self.news_sentiment:
            news_score = self.news_sentiment.get('avg_sentiment', 0)
            # Normalize to -1 to 1 range
            news_score = max(-1, min(1, news_score * 2))
            scores.append(news_score)
            weights.append(0.35)
        
        # Social sentiment (weight: 30%)
        if self.social_sentiment:
            social_score = self.social_sentiment.get('overall_score', 0)
            scores.append(social_score)
            weights.append(0.30)
        
        # Insider activity (weight: 20%)
        if self.insider_data:
            insider_score = self.insider_data.get('insider_score', 0)
            scores.append(insider_score)
            weights.append(0.20)
        
        # Analyst ratings (weight: 15%)
        if self.analyst_data:
            analyst_score = self.analyst_data.get('analyst_score', 0)
            scores.append(analyst_score)
            weights.append(0.15)
        
        # Calculate weighted average
        if scores:
            total_weight = sum(weights)
            self.score = sum(s * w for s, w in zip(scores, weights)) / total_weight
        else:
            self.score = 0
        
        # Determine recommendation
        if self.score > 0.2:
            recommendation = 'BULLISH'
        elif self.score < -0.2:
            recommendation = 'BEARISH'
        else:
            recommendation = 'NEUTRAL'
        
        bullish_count = len(self.signals.get('bullish', []))
        bearish_count = len(self.signals.get('bearish', []))
        
        return {
            'score': self.score,
            'recommendation': recommendation,
            'bullish_signals': bullish_count,
            'bearish_signals': bearish_count,
            'neutral_signals': len(self.signals.get('neutral', [])),
            'signals': self.signals,
            'component_scores': {
                'news': self.news_sentiment.get('avg_sentiment', 0) if self.news_sentiment else 0,
                'social': self.social_sentiment.get('overall_score', 0) if self.social_sentiment else 0,
                'insider': self.insider_data.get('insider_score', 0) if self.insider_data else 0,
                'analyst': self.analyst_data.get('analyst_score', 0) if self.analyst_data else 0,
            }
        }
    
    def analyze(self) -> dict:
        """Run complete sentiment analysis."""
        # Gather all sentiment data
        self.analyze_news_sentiment()
        self.analyze_social_sentiment()
        self.insider_data = self.fetch_insider_transactions()
        self.analyst_data = self.fetch_analyst_ratings()
        
        # Generate signals and score
        self.generate_signals()
        score_data = self.calculate_score()
        
        return {
            'ticker': self.ticker,
            'company_name': self.company_name,
            'analysis_type': 'sentiment',
            'weight': 0.4,
            'news_sentiment': self.news_sentiment,
            'social_sentiment': self.social_sentiment,
            'insider_data': self.insider_data,
            'analyst_data': self.analyst_data,
            'score_data': score_data,
        }


if __name__ == "__main__":
    # Test the module
    analyzer = SentimentAnalyzer("AAPL")
    result = analyzer.analyze()
    print(f"Sentiment Analysis for {result['ticker']}")
    print(f"Score: {result['score_data']['score']:.2f}")
    print(f"Recommendation: {result['score_data']['recommendation']}")
    print(f"\nNews Articles: {result['news_sentiment'].get('total_count', 0)}")
    print(f"News Sentiment: {result['news_sentiment'].get('sentiment_label', 'N/A')}")
    print(f"\nBullish Signals: {result['score_data']['signals']['bullish']}")
    print(f"Bearish Signals: {result['score_data']['signals']['bearish']}")
