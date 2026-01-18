"""
Fundamental Analysis Module
Analyzes company financials, valuation metrics, and growth indicators.
Weight: 20% of total score
"""

import yfinance as yf
import pandas as pd
import numpy as np
from typing import Dict, Optional
from datetime import datetime


class FundamentalAnalyzer:
    """Performs fundamental analysis on stock data."""
    
    def __init__(self, ticker: str):
        self.ticker = ticker.upper()
        self.stock = yf.Ticker(self.ticker)
        self.info = {}
        self.financials = {}
        self.valuation = {}
        self.growth = {}
        self.profitability = {}
        self.financial_health = {}
        self.signals = {}
        self.score = 0
        
    def fetch_company_info(self) -> dict:
        """Fetch basic company information."""
        try:
            self.info = self.stock.info
            return {
                'name': self.info.get('shortName', self.ticker),
                'sector': self.info.get('sector', 'N/A'),
                'industry': self.info.get('industry', 'N/A'),
                'market_cap': self.info.get('marketCap', 0),
                'employees': self.info.get('fullTimeEmployees', 0),
                'website': self.info.get('website', ''),
                'description': self.info.get('longBusinessSummary', '')[:500] + '...' if self.info.get('longBusinessSummary') else '',
                'country': self.info.get('country', 'N/A'),
                'exchange': self.info.get('exchange', 'N/A'),
            }
        except Exception as e:
            print(f"Error fetching company info: {e}")
            return {}
    
    def analyze_valuation(self) -> dict:
        """Analyze valuation metrics."""
        try:
            pe_ratio = self.info.get('trailingPE')
            forward_pe = self.info.get('forwardPE')
            peg_ratio = self.info.get('pegRatio')
            pb_ratio = self.info.get('priceToBook')
            ps_ratio = self.info.get('priceToSalesTrailing12Months')
            ev_ebitda = self.info.get('enterpriseToEbitda')
            ev_revenue = self.info.get('enterpriseToRevenue')
            
            # Get sector averages (approximate)
            sector = self.info.get('sector', '')
            sector_pe_avg = self._get_sector_pe_average(sector)
            
            self.valuation = {
                'pe_ratio': pe_ratio,
                'forward_pe': forward_pe,
                'peg_ratio': peg_ratio,
                'pb_ratio': pb_ratio,
                'ps_ratio': ps_ratio,
                'ev_ebitda': ev_ebitda,
                'ev_revenue': ev_revenue,
                'sector_pe_avg': sector_pe_avg,
                'pe_vs_sector': 'undervalued' if pe_ratio and sector_pe_avg and pe_ratio < sector_pe_avg else 'overvalued',
                'peg_assessment': self._assess_peg(peg_ratio),
            }
            
            return self.valuation
        except Exception as e:
            print(f"Error analyzing valuation: {e}")
            return {}
    
    def _get_sector_pe_average(self, sector: str) -> float:
        """Get approximate sector P/E average."""
        sector_pe = {
            'Technology': 28,
            'Healthcare': 22,
            'Financial Services': 15,
            'Consumer Cyclical': 20,
            'Consumer Defensive': 22,
            'Industrials': 20,
            'Energy': 12,
            'Utilities': 18,
            'Real Estate': 35,
            'Basic Materials': 15,
            'Communication Services': 18,
        }
        return sector_pe.get(sector, 20)
    
    def _assess_peg(self, peg: float) -> str:
        """Assess PEG ratio."""
        if peg is None:
            return 'N/A'
        if peg < 1:
            return 'undervalued'
        elif peg < 2:
            return 'fairly_valued'
        else:
            return 'overvalued'
    
    def analyze_growth(self) -> dict:
        """Analyze growth metrics."""
        try:
            revenue_growth = self.info.get('revenueGrowth')
            earnings_growth = self.info.get('earningsGrowth')
            earnings_quarterly_growth = self.info.get('earningsQuarterlyGrowth')
            
            # Get historical financials for trend analysis
            income_stmt = self.stock.income_stmt
            
            revenue_cagr = None
            earnings_cagr = None
            
            if income_stmt is not None and not income_stmt.empty:
                try:
                    revenues = income_stmt.loc['Total Revenue'] if 'Total Revenue' in income_stmt.index else None
                    if revenues is not None and len(revenues) >= 2:
                        latest = revenues.iloc[0]
                        oldest = revenues.iloc[-1]
                        years = len(revenues) - 1
                        if oldest > 0 and years > 0:
                            revenue_cagr = ((latest / oldest) ** (1 / years) - 1) * 100
                    
                    net_income = income_stmt.loc['Net Income'] if 'Net Income' in income_stmt.index else None
                    if net_income is not None and len(net_income) >= 2:
                        latest = net_income.iloc[0]
                        oldest = net_income.iloc[-1]
                        years = len(net_income) - 1
                        if oldest > 0 and years > 0:
                            earnings_cagr = ((latest / oldest) ** (1 / years) - 1) * 100
                except Exception:
                    pass
            
            self.growth = {
                'revenue_growth_yoy': revenue_growth * 100 if revenue_growth else None,
                'earnings_growth_yoy': earnings_growth * 100 if earnings_growth else None,
                'earnings_quarterly_growth': earnings_quarterly_growth * 100 if earnings_quarterly_growth else None,
                'revenue_cagr': revenue_cagr,
                'earnings_cagr': earnings_cagr,
                'growth_assessment': self._assess_growth(revenue_growth, earnings_growth),
            }
            
            return self.growth
        except Exception as e:
            print(f"Error analyzing growth: {e}")
            return {}
    
    def _assess_growth(self, revenue_growth: float, earnings_growth: float) -> str:
        """Assess overall growth."""
        if revenue_growth is None and earnings_growth is None:
            return 'N/A'
        
        avg_growth = 0
        count = 0
        if revenue_growth:
            avg_growth += revenue_growth
            count += 1
        if earnings_growth:
            avg_growth += earnings_growth
            count += 1
        
        if count > 0:
            avg_growth /= count
        
        if avg_growth > 0.2:
            return 'high_growth'
        elif avg_growth > 0.1:
            return 'moderate_growth'
        elif avg_growth > 0:
            return 'low_growth'
        else:
            return 'declining'
    
    def analyze_profitability(self) -> dict:
        """Analyze profitability metrics."""
        try:
            gross_margin = self.info.get('grossMargins')
            operating_margin = self.info.get('operatingMargins')
            profit_margin = self.info.get('profitMargins')
            roe = self.info.get('returnOnEquity')
            roa = self.info.get('returnOnAssets')
            
            self.profitability = {
                'gross_margin': gross_margin * 100 if gross_margin else None,
                'operating_margin': operating_margin * 100 if operating_margin else None,
                'profit_margin': profit_margin * 100 if profit_margin else None,
                'roe': roe * 100 if roe else None,
                'roa': roa * 100 if roa else None,
                'profitability_assessment': self._assess_profitability(profit_margin, roe),
            }
            
            return self.profitability
        except Exception as e:
            print(f"Error analyzing profitability: {e}")
            return {}
    
    def _assess_profitability(self, profit_margin: float, roe: float) -> str:
        """Assess overall profitability."""
        if profit_margin is None and roe is None:
            return 'N/A'
        
        score = 0
        if profit_margin:
            if profit_margin > 0.15:
                score += 2
            elif profit_margin > 0.05:
                score += 1
            elif profit_margin < 0:
                score -= 1
        
        if roe:
            if roe > 0.15:
                score += 2
            elif roe > 0.10:
                score += 1
            elif roe < 0:
                score -= 1
        
        if score >= 3:
            return 'excellent'
        elif score >= 1:
            return 'good'
        elif score >= 0:
            return 'average'
        else:
            return 'poor'
    
    def analyze_financial_health(self) -> dict:
        """Analyze financial health metrics."""
        try:
            current_ratio = self.info.get('currentRatio')
            quick_ratio = self.info.get('quickRatio')
            debt_to_equity = self.info.get('debtToEquity')
            total_debt = self.info.get('totalDebt')
            total_cash = self.info.get('totalCash')
            free_cash_flow = self.info.get('freeCashflow')
            operating_cash_flow = self.info.get('operatingCashflow')
            
            # Calculate net debt
            net_debt = (total_debt - total_cash) if total_debt and total_cash else None
            
            self.financial_health = {
                'current_ratio': current_ratio,
                'quick_ratio': quick_ratio,
                'debt_to_equity': debt_to_equity,
                'total_debt': total_debt,
                'total_cash': total_cash,
                'net_debt': net_debt,
                'free_cash_flow': free_cash_flow,
                'operating_cash_flow': operating_cash_flow,
                'health_assessment': self._assess_financial_health(current_ratio, debt_to_equity, free_cash_flow),
            }
            
            return self.financial_health
        except Exception as e:
            print(f"Error analyzing financial health: {e}")
            return {}
    
    def _assess_financial_health(self, current_ratio: float, debt_to_equity: float, fcf: float) -> str:
        """Assess overall financial health."""
        score = 0
        
        if current_ratio:
            if current_ratio > 2:
                score += 2
            elif current_ratio > 1:
                score += 1
            else:
                score -= 1
        
        if debt_to_equity:
            if debt_to_equity < 50:
                score += 2
            elif debt_to_equity < 100:
                score += 1
            elif debt_to_equity > 200:
                score -= 1
        
        if fcf:
            if fcf > 0:
                score += 2
            else:
                score -= 1
        
        if score >= 4:
            return 'strong'
        elif score >= 2:
            return 'healthy'
        elif score >= 0:
            return 'moderate'
        else:
            return 'weak'
    
    def analyze_dividends(self) -> dict:
        """Analyze dividend metrics."""
        try:
            dividend_yield = self.info.get('dividendYield')
            dividend_rate = self.info.get('dividendRate')
            payout_ratio = self.info.get('payoutRatio')
            five_year_avg_yield = self.info.get('fiveYearAvgDividendYield')
            
            return {
                'dividend_yield': dividend_yield * 100 if dividend_yield else 0,
                'dividend_rate': dividend_rate,
                'payout_ratio': payout_ratio * 100 if payout_ratio else None,
                'five_year_avg_yield': five_year_avg_yield,
                'dividend_assessment': self._assess_dividend(dividend_yield, payout_ratio),
            }
        except Exception as e:
            print(f"Error analyzing dividends: {e}")
            return {}
    
    def _assess_dividend(self, yield_val: float, payout_ratio: float) -> str:
        """Assess dividend quality."""
        if not yield_val or yield_val == 0:
            return 'no_dividend'
        
        if payout_ratio:
            if payout_ratio > 0.8:
                return 'unsustainable'
            elif payout_ratio > 0.6:
                return 'high_payout'
            elif payout_ratio > 0.3:
                return 'sustainable'
            else:
                return 'low_payout'
        
        return 'dividend_paying'
    
    def generate_signals(self) -> dict:
        """Generate fundamental signals."""
        signals = {
            'bullish': [],
            'bearish': [],
            'neutral': []
        }
        
        # Valuation signals
        if self.valuation:
            peg = self.valuation.get('peg_ratio')
            if peg and peg < 1:
                signals['bullish'].append(f"PEG ratio undervalued ({peg:.2f})")
            elif peg and peg > 2:
                signals['bearish'].append(f"PEG ratio overvalued ({peg:.2f})")
            
            pe = self.valuation.get('pe_ratio')
            sector_pe = self.valuation.get('sector_pe_avg')
            if pe and sector_pe:
                if pe < sector_pe * 0.8:
                    signals['bullish'].append(f"P/E below sector average ({pe:.1f} vs {sector_pe:.1f})")
                elif pe > sector_pe * 1.3:
                    signals['bearish'].append(f"P/E above sector average ({pe:.1f} vs {sector_pe:.1f})")
        
        # Growth signals
        if self.growth:
            rev_growth = self.growth.get('revenue_growth_yoy')
            if rev_growth:
                if rev_growth > 20:
                    signals['bullish'].append(f"Strong revenue growth ({rev_growth:.1f}%)")
                elif rev_growth < 0:
                    signals['bearish'].append(f"Revenue declining ({rev_growth:.1f}%)")
            
            earn_growth = self.growth.get('earnings_growth_yoy')
            if earn_growth:
                if earn_growth > 20:
                    signals['bullish'].append(f"Strong earnings growth ({earn_growth:.1f}%)")
                elif earn_growth < 0:
                    signals['bearish'].append(f"Earnings declining ({earn_growth:.1f}%)")
        
        # Profitability signals
        if self.profitability:
            roe = self.profitability.get('roe')
            if roe:
                if roe > 20:
                    signals['bullish'].append(f"High ROE ({roe:.1f}%)")
                elif roe < 5:
                    signals['bearish'].append(f"Low ROE ({roe:.1f}%)")
            
            margin = self.profitability.get('profit_margin')
            if margin:
                if margin > 15:
                    signals['bullish'].append(f"High profit margin ({margin:.1f}%)")
                elif margin < 0:
                    signals['bearish'].append(f"Negative profit margin ({margin:.1f}%)")
        
        # Financial health signals
        if self.financial_health:
            de = self.financial_health.get('debt_to_equity')
            if de:
                if de < 30:
                    signals['bullish'].append(f"Low debt-to-equity ({de:.1f})")
                elif de > 150:
                    signals['bearish'].append(f"High debt-to-equity ({de:.1f})")
            
            fcf = self.financial_health.get('free_cash_flow')
            if fcf:
                if fcf > 0:
                    signals['bullish'].append("Positive free cash flow")
                else:
                    signals['bearish'].append("Negative free cash flow")
            
            current = self.financial_health.get('current_ratio')
            if current:
                if current > 2:
                    signals['bullish'].append(f"Strong current ratio ({current:.2f})")
                elif current < 1:
                    signals['bearish'].append(f"Weak current ratio ({current:.2f})")
        
        self.signals = signals
        return signals
    
    def calculate_score(self) -> dict:
        """Calculate fundamental analysis score (-1 to +1)."""
        scores = []
        weights = []
        
        # Valuation score (weight: 30%)
        if self.valuation:
            val_score = 0
            peg = self.valuation.get('peg_ratio')
            if peg:
                if peg < 1:
                    val_score += 0.5
                elif peg > 2:
                    val_score -= 0.5
            
            pe_assessment = self.valuation.get('pe_vs_sector')
            if pe_assessment == 'undervalued':
                val_score += 0.5
            elif pe_assessment == 'overvalued':
                val_score -= 0.5
            
            scores.append(max(-1, min(1, val_score)))
            weights.append(0.30)
        
        # Growth score (weight: 30%)
        if self.growth:
            growth_assessment = self.growth.get('growth_assessment')
            growth_score = {
                'high_growth': 1,
                'moderate_growth': 0.5,
                'low_growth': 0,
                'declining': -0.5,
                'N/A': 0
            }.get(growth_assessment, 0)
            scores.append(growth_score)
            weights.append(0.30)
        
        # Profitability score (weight: 20%)
        if self.profitability:
            prof_assessment = self.profitability.get('profitability_assessment')
            prof_score = {
                'excellent': 1,
                'good': 0.5,
                'average': 0,
                'poor': -0.5,
                'N/A': 0
            }.get(prof_assessment, 0)
            scores.append(prof_score)
            weights.append(0.20)
        
        # Financial health score (weight: 20%)
        if self.financial_health:
            health_assessment = self.financial_health.get('health_assessment')
            health_score = {
                'strong': 1,
                'healthy': 0.5,
                'moderate': 0,
                'weak': -0.5,
            }.get(health_assessment, 0)
            scores.append(health_score)
            weights.append(0.20)
        
        # Calculate weighted average
        if scores:
            total_weight = sum(weights)
            self.score = sum(s * w for s, w in zip(scores, weights)) / total_weight
        else:
            self.score = 0
        
        # Determine recommendation
        if self.score > 0.3:
            recommendation = 'BULLISH'
        elif self.score < -0.3:
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
                'valuation': self.valuation.get('peg_assessment', 'N/A'),
                'growth': self.growth.get('growth_assessment', 'N/A'),
                'profitability': self.profitability.get('profitability_assessment', 'N/A'),
                'financial_health': self.financial_health.get('health_assessment', 'N/A'),
            }
        }
    
    def analyze(self) -> dict:
        """Run complete fundamental analysis."""
        company_info = self.fetch_company_info()
        self.analyze_valuation()
        self.analyze_growth()
        self.analyze_profitability()
        self.analyze_financial_health()
        dividends = self.analyze_dividends()
        
        self.generate_signals()
        score_data = self.calculate_score()
        
        return {
            'ticker': self.ticker,
            'analysis_type': 'fundamental',
            'weight': 0.2,
            'company_info': company_info,
            'valuation': self.valuation,
            'growth': self.growth,
            'profitability': self.profitability,
            'financial_health': self.financial_health,
            'dividends': dividends,
            'score_data': score_data,
        }


if __name__ == "__main__":
    # Test the module
    analyzer = FundamentalAnalyzer("AAPL")
    result = analyzer.analyze()
    print(f"Fundamental Analysis for {result['ticker']}")
    print(f"Company: {result['company_info'].get('name', 'N/A')}")
    print(f"Sector: {result['company_info'].get('sector', 'N/A')}")
    print(f"\nScore: {result['score_data']['score']:.2f}")
    print(f"Recommendation: {result['score_data']['recommendation']}")
    print(f"\nValuation: P/E={result['valuation'].get('pe_ratio', 'N/A')}, PEG={result['valuation'].get('peg_ratio', 'N/A')}")
    print(f"Growth: {result['growth'].get('growth_assessment', 'N/A')}")
    print(f"Profitability: {result['profitability'].get('profitability_assessment', 'N/A')}")
    print(f"Financial Health: {result['financial_health'].get('health_assessment', 'N/A')}")
    print(f"\nBullish Signals: {result['score_data']['signals']['bullish']}")
    print(f"Bearish Signals: {result['score_data']['signals']['bearish']}")
