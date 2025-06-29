import streamlit as st
import datetime
from collections import defaultdict, Counter
import json
import requests
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import numpy as np
import pytz
import ta  # Technical analysis library
import re  # Added for better text processing

# Configure plot style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("Set2")
st.set_page_config(layout="wide", page_title="Stock News Analytics Dashboard", page_icon="📈")

# Define event types with keywords
EVENT_TYPES = [
    ("acquisition", ["acquire", "acquisition", "buyout", "takeover", "merger", "purchase", 
                     "stake buy", "stake sale", "absorb", "consolidat"]),
    ("partnership", ["partner", "alliance", "collaborat", "joint venture", "joins hands", 
                     "teams up", "strategic tie-up", "cooperat", "synergy"]),
    ("agreement", ["agreement", "contract", "pact", "deal", "signs", "memorandum", "mou", 
                   "accord", "settlement", "understanding"]),
    ("investment", ["invest", "funding", "raise capital", "infuse", "fundraise", "vc funding", 
                    "private equity", "capital infusion", "series a", "series b"]),
    ("launch", ["launch", "introduce", "release", "unveil", "debut", "premiere", "roll out", 
                "commence", "inaugurate", "begin"]),
    ("expansion", ["expand", "expansion", "new facility", "new plant", "new office", "growth", 
                   "geographic", "capacity increase", "scale up", "broaden"]),
    ("award", ["award", "prize", "recognition", "honor", "achievement", "accolade", "medal", 
               "trophy", "distinction", "commendation"]),
    ("leadership", ["appoint", "hire", "resign", "exit", "join as", "takes over", "ceo", "cfo", 
                    "cto", "board", "director", "management", "executive", "promote"]),
    ("financial", ["results", "earnings", "profit", "revenue", "dividend", "q1", "q2", "q3", "q4", 
                   "quarterly", "annual", "balance sheet", "p&l", "financials"]),
    ("regulatory", ["regulator", "sebi", "rbi", "government", "approval", "clearance", "compliance", 
                    "investigation", "penalty", "lawsuit", "court", "legal", "settlement"])
]

# Define event duration classifications
EVENT_DURATION = {
    "acquisition": "long-term",
    "partnership": "long-term",
    "agreement": "medium-term",
    "investment": "long-term",
    "launch": "short-term",
    "expansion": "long-term",
    "award": "short-term",
    "leadership": "medium-term",
    "financial": "short-term",
    "regulatory": "medium-term",
    "other": "unknown"
}

# Supply chain relationships
SUPPLY_CHAIN = {
    "AUTO": ["TATASTEEL", "BOSCHLTD", "MOTHERSON", "BAJAJ-AUTO", "MARUTI"],
    "PHARMA": ["LAURUSLABS", "SOLARA", "SUNPHARMA", "DRREDDY"],
    "IT": ["INFY", "TCS", "WIPRO", "HCLTECH"],
    "BANKING": ["HDFCBANK", "ICICIBANK", "KOTAKBANK", "SBIN"],
    "ENERGY": ["RELIANCE", "ONGC", "GAIL", "IOC"]
}

# Cache news data to avoid repeated API calls
@st.cache_data(ttl=3600)  # Refresh every hour
def fetch_news_data():
    url = "https://service.upstox.com/content/open/v5/news/sub-category/news/list//market-news/stocks?page=1&pageSize=500"
    response = requests.get(url)
    return response.json()['data']

# Cache stock data with extended period for technical analysis
@st.cache_data(ttl=3600)
def get_stock_data(symbol, exchange):
    suffix_map = {'nse': '.NS', 'bse': '.BO'}
    suffix = suffix_map.get(exchange.lower(), '')
    ticker = symbol + suffix
    
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="6mo")
        if hist.empty:
            return None, None, None, None
        
        start_price = hist['Close'].iloc[0]
        current_price = hist['Close'].iloc[-1]
        pct_change = ((current_price - start_price) / start_price) * 100
        
        if pct_change > 5:
            trend = "strong_up"
        elif pct_change > 1:
            trend = "moderate_up"
        elif pct_change < -5:
            trend = "strong_down"
        elif pct_change < -1:
            trend = "moderate_down"
        else:
            trend = "neutral"
            
        return current_price, pct_change, trend, hist
    
    except Exception as e:
        return None, None, None, None

# Cache sector data
@st.cache_data(ttl=86400)
def get_sector(symbol, exchange):
    suffix_map = {'nse': '.NS', 'bse': '.BO'}
    suffix = suffix_map.get(exchange.lower(), '')
    ticker = symbol + suffix
    
    try:
        stock = yf.Ticker(ticker)
        sector = stock.info.get('sector', 'Unknown')
        return sector
    except:
        return "Unknown"

# Improved event extraction with regex matching
def extract_event_type(headline):
    headline_lower = headline.lower()
    # Clean and tokenize the headline
    tokens = re.findall(r'\b\w+\b', headline_lower)
    
    for event_type, keywords in EVENT_TYPES:
        for kw in keywords:
            # Check for keyword in tokens (whole word match)
            if any(kw in token for token in tokens):
                return event_type
    return "other"

# Enhanced sentiment analysis with negation handling
def compute_sentiment_score(headline):
    headline_lower = headline.lower()
    
    # Handle negations
    negation_words = ['not', 'no', 'never', 'none', 'nobody', 'nothing', 'neither', 'nor', 'nowhere']
    positive_words = ['growth', 'profit', 'gain', 'surge', 'rise', 'upgrade', 'buy', 'strong', 
                      'win', 'success', 'beat', 'positive', 'high', 'increase', 'outperform']
    negative_words = ['fall', 'loss', 'decline', 'drop', 'cut', 'sell', 'weak', 'risk', 'warn', 
                      'fail', 'negative', 'low', 'decrease', 'underperform', 'downgrade']
    
    # Tokenize with context for negation
    tokens = headline_lower.split()
    score = 0
    
    for i, token in enumerate(tokens):
        if token in positive_words:
            # Check for negation in previous words
            if i > 0 and tokens[i-1] in negation_words:
                score -= 1  # Positive word negated becomes negative
            else:
                score += 1
                
        elif token in negative_words:
            # Check for negation in previous words
            if i > 0 and tokens[i-1] in negation_words:
                score += 1  # Negative word negated becomes positive
            else:
                score -= 1
                
    return score

# Event type extraction
def extract_event_type(headline):
    headline_lower = headline.lower()
    for event_type, keywords in EVENT_TYPES:
        for kw in keywords:
            if kw in headline_lower:
                return event_type
    return "other"

# Enhanced correlation analysis with sentiment weighting
def analyze_stock_news_correlation(events, history):
    if history is None:
        return 0
        
    # Correct unpacking of 4 elements per event
    event_dates = [date for _, date, _, _ in events]
    prices = history[['Close']].reset_index()
    
    results = []
    for event_type, date, headline, sentiment in events:
        # Convert to date only for comparison
        event_date = date.date()
        
        # Find previous trading day
        prev_days = prices[prices['Date'].dt.date < event_date]
        if len(prev_days) < 1:
            continue
            
        prev_day = prev_days['Date'].iloc[-1]
        prev_price = prices[prices['Date'] == prev_day]['Close'].values
        
        # Find next trading day
        next_days = prices[prices['Date'].dt.date > event_date]
        if len(next_days) < 1:
            continue
            
        next_day = next_days['Date'].iloc[0]
        next_price = prices[prices['Date'] == next_day]['Close'].values
        
        if len(prev_price) > 0 and len(next_price) > 0:
            change = ((next_price[0] - prev_price[0]) / prev_price[0]) * 100
            weighted_change = change * (1 + 0.2 * np.sign(sentiment))
            results.append(weighted_change)
    
    if results:
        return sum(results) / len(results)
    return 0

# Event clustering detection
def detect_event_clusters(events):
    if len(events) < 3:
        return 0
        
    # Extract dates and sort them
    event_dates = sorted([date for _, date, _, _ in events])
    clusters = 0
    i = 0
    
    while i < len(event_dates) - 2:
        if (event_dates[i+2] - event_dates[i]).days <= 7:
            clusters += 1
            i += 3  # Skip events in current cluster
        else:
            i += 1
            
    return clusters

# Volume spike analysis
def detect_volume_spikes(events, history):
    if history is None or history.empty:
        return 0
    
    volume_spikes = 0
    prices = history[['Volume']].copy()
    prices.reset_index(inplace=True)
    
    # Convert to date for comparison
    prices['Date'] = prices['Date'].dt.date
    
    for _, date, _, _ in events:
        event_date = date.date()
        
        # Check if event date is in trading data
        if event_date not in prices['Date'].values:
            continue
            
        # Get previous trading days
        prev_days = prices[prices['Date'] < event_date]
        if len(prev_days) < 5:
            continue
            
        # Calculate average volume of previous 5 trading days
        avg_volume = prev_days['Volume'].tail(5).mean()
        
        # Get volume on event day
        event_volume = prices[prices['Date'] == event_date]['Volume'].values[0]
        
        if event_volume > 1.5 * avg_volume:
            volume_spikes += 1
            
    return volume_spikes

# Regulatory risk scoring
def regulatory_risk_score(company_news):
    reg_keywords = ["sebi", "rbi", "probe", "penalty", "investigation", "fine", "lawsuit"]
    count = sum(1 for _, _, headline, _ in company_news 
               if any(kw in headline.lower() for kw in reg_keywords))
    return min(count * 20, 100)  # Scale to 0-100

# Trend score calculation with technical indicators
def calculate_trend_score(history, sentiment):
    if history is None or len(history) < 30:
        return 50  # Neutral score
    
    try:
        # Calculate technical indicators
        history['SMA_20'] = history['Close'].rolling(window=20).mean()
        history['SMA_50'] = history['Close'].rolling(window=50).mean()
        
        # RSI calculation
        delta = history['Close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        # MACD calculation
        ema12 = history['Close'].ewm(span=12, adjust=False).mean()
        ema26 = history['Close'].ewm(span=26, adjust=False).mean()
        macd = ema12 - ema26
        signal = macd.ewm(span=9, adjust=False).mean()
        
        # Current values
        current_rsi = rsi.iloc[-1]
        macd_diff = macd.iloc[-1] - signal.iloc[-1]
        
        # Scoring system
        score = 50  # Base score
        
        # RSI scoring
        if current_rsi < 30:
            score += 15  # Oversold - bullish
        elif current_rsi > 70:
            score -= 15  # Overbought - bearish
        
        # MACD scoring
        if macd_diff > 0:
            score += 10
        else:
            score -= 10
        
        # Moving averages scoring
        if history['SMA_20'].iloc[-1] > history['SMA_50'].iloc[-1]:
            score += 10  # Golden cross
        else:
            score -= 10  # Death cross
        
        # Sentiment adjustment
        score += sentiment * 5
        
        # Ensure score is between 0-100
        return max(0, min(100, score))
    except Exception as e:
        return 50  # Return neutral on error

# Supply chain impact analysis
def analyze_supply_chain_impact(sector, events, news_data):
    if sector not in SUPPLY_CHAIN:
        return 0
    
    related_companies = SUPPLY_CHAIN[sector]
    impact_score = 0
    
    for event_type, date, _, _ in events:
        # Check for news about related companies around the same time
        related_events = [
            item for item in news_data 
            if 'linkedScrips' in item and 
            any(scrip['symbol'] in related_companies 
                for scrip in item['linkedScrips'])
        ]
        
        for item in related_events:
            try:
                pub_date = datetime.datetime.strptime(
                    item['publishedAt'].replace('Z', ''),
                    "%Y-%m-%dT%H:%M:%S.%f"
                )
                if abs((pub_date - date).days) <= 3:
                    impact_score += 10
            except:
                continue
                
    return min(impact_score, 100)

# Generate word cloud
def generate_word_cloud(news_data):
    headlines = [item['headline'] for item in news_data]
    text = " ".join(headlines)
    
    wordcloud = WordCloud(width=800, height=400, 
                          background_color='white',
                          colormap='viridis',
                          max_words=100).generate(text)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis("off")
    return fig

# Generate sentiment analysis pie chart
def generate_sentiment_analysis(news_data):
    sentiments = []
    for item in news_data:
        headline = item['headline'].lower()
        positive_count = sum(headline.count(word) for word in 
                           ['growth','profit','gain','surge','rise','upgrade','buy','strong','win','success'])
        negative_count = sum(headline.count(word) for word in 
                           ['fall','loss','decline','drop','cut','sell','weak','risk','warn','fail'])
        
        if positive_count > negative_count:
            sentiments.append('Positive')
        elif negative_count > positive_count:
            sentiments.append('Negative')
        else:
            sentiments.append('Neutral')
    
    sentiment_counts = pd.Series(sentiments).value_counts()
    fig = px.pie(sentiment_counts, values=sentiment_counts.values, 
                 names=sentiment_counts.index, 
                 title='News Sentiment Distribution',
                 color_discrete_sequence=px.colors.sequential.Viridis)
    return fig

# Generate top companies chart
def generate_top_companies_chart(news_data):
    companies = []
    for item in news_data:
        if 'linkedScrips' in item:
            for scrip in item['linkedScrips']:
                companies.append(scrip['symbol'])
    
    if not companies:
        return None
    
    company_counts = pd.Series(companies).value_counts().head(10)
    fig = px.bar(company_counts, x=company_counts.values, y=company_counts.index,
                 orientation='h', title='Top Companies in News',
                 color=company_counts.values,
                 color_continuous_scale='Viridis')
    fig.update_layout(yaxis_title='Company', xaxis_title='News Count')
    return fig

# Generate event timeline
def generate_event_timeline(news_data):
    events = []
    for item in news_data:
        try:
            date = datetime.datetime.strptime(
                item['publishedAt'].replace('Z', ''),
                "%Y-%m-%dT%H:%M:%S.%f"
            ).date()
            events.append({'date': date, 'headline': item['headline']})
        except:
            continue
    
    if not events:
        return None
    
    events_df = pd.DataFrame(events)
    events_df = events_df.groupby('date').size().reset_index(name='count')
    
    fig = px.line(events_df, x='date', y='count', 
                  title='News Events Timeline',
                  markers=True)
    fig.update_traces(line_color='#5e35b1', marker_color='#9c27b0')
    return fig

# Generate sector performance heatmap
def generate_sector_heatmap(news_data):
    sectors = defaultdict(list)
    
    for item in news_data:
        if 'linkedScrips' in item:
            for scrip in item['linkedScrips']:
                symbol = scrip['symbol']
                exchange = scrip['exchange']
                sector = get_sector(symbol, exchange)
                if sector != "Unknown":
                    sectors[sector].append(symbol)
    
    if not sectors:
        return None
    
    # Get sector performance
    sector_perf = {}
    for sector, symbols in sectors.items():
        perf = []
        for symbol in set(symbols):  # Deduplicate
            # Get exchange from first occurrence
            exchange = next((scrip['exchange'] for item in news_data 
                            if 'linkedScrips' in item 
                            for scrip in item['linkedScrips'] 
                            if scrip['symbol'] == symbol), 'nse')
            
            _, pct_change, _, _ = get_stock_data(symbol, exchange)
            if pct_change is not None:
                perf.append(pct_change)
        if perf:
            sector_perf[sector] = sum(perf) / len(perf)
    
    if not sector_perf:
        return None
    
    df = pd.DataFrame(list(sector_perf.items()), columns=['Sector', 'Performance'])
    df = df.sort_values('Performance', ascending=False)
    
    fig = px.bar(df, x='Sector', y='Performance', 
                 title='Sector Performance (Avg 3M Return)',
                 color='Performance',
                 color_continuous_scale='Viridis')
    fig.update_layout(xaxis_title='', yaxis_title='Average Return (%)')
    return fig

# Generate regulatory risk gauge
def generate_regulatory_gauge(risk_score):
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = risk_score,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Regulatory Risk Score"},
        gauge = {
            'axis': {'range': [0, 100]},
            'steps': [
                {'range': [0, 30], 'color': "lightgreen"},
                {'range': [30, 70], 'color': "yellow"},
                {'range': [70, 100], 'color': "red"}],
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.75,
                'value': risk_score}}))
    return fig

def main():
    # Load news data
    st.sidebar.title("Dashboard Settings")
    st.sidebar.info("Advanced stock news analytics with predictive capabilities")
    
    # Add model configuration
    st.sidebar.subheader("Prediction Model Settings")
    sentiment_weight = st.sidebar.slider("Sentiment Weight", 0.1, 0.5, 0.2)
    min_confidence = st.sidebar.slider("Minimum Confidence", 30, 90, 50)
    
    # Load data with progress indicator
    with st.spinner('Fetching latest news data...'):
        news_data = fetch_news_data()
    
    st.title("📈 Advanced Stock News Analytics Dashboard")
    st.caption("Predictive analytics platform for corporate events and stock movements")
    
    # Overview metrics
    st.subheader("Market Overview")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total News Articles", len(news_data))
    unique_companies = len(set(scrip['symbol'] for item in news_data if 'linkedScrips' in item for scrip in item['linkedScrips']))
    col2.metric("Companies Covered", unique_companies)
    
    # Get Nifty and Sensex data
    try:
        nifty = yf.Ticker("^NSEI")
        nifty_data = nifty.history(period='1d')
        nifty_change = ((nifty_data['Close'][0] - nifty_data['Open'][0]) / nifty_data['Open'][0]) * 100
        col3.metric("Nifty 50", f"{nifty_data['Close'][0]:.2f}", f"{nifty_change:.2f}%")
    except:
        col3.metric("Nifty 50", "N/A", "N/A")
    
    try:
        sensex = yf.Ticker("^BSESN")
        sensex_data = sensex.history(period='1d')
        sensex_change = ((sensex_data['Close'][0] - sensex_data['Open'][0]) / sensex_data['Open'][0]) * 100
        col4.metric("BSE Sensex", f"{sensex_data['Close'][0]:.2f}", f"{sensex_change:.2f}%")
    except:
        col4.metric("BSE Sensex", "N/A", "N/A")
    
    # Top row visualizations
    st.subheader("News Analytics")
    col1, col2 = st.columns(2)
    
    with col1:
        st.pyplot(generate_word_cloud(news_data))
        st.caption("Word Cloud of Most Frequent Terms in News Headlines")
    
    with col2:
        st.plotly_chart(generate_sentiment_analysis(news_data), use_container_width=True)
    
    # Middle row visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        top_companies = generate_top_companies_chart(news_data)
        if top_companies:
            st.plotly_chart(top_companies, use_container_width=True)
        else:
            st.warning("No company data available for visualization")
    
    with col2:
        timeline = generate_event_timeline(news_data)
        if timeline:
            st.plotly_chart(timeline, use_container_width=True)
        else:
            st.warning("No date data available for timeline")
    
    # Third row visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        sector_heatmap = generate_sector_heatmap(news_data)
        if sector_heatmap:
            st.plotly_chart(sector_heatmap, use_container_width=True)
        else:
            st.warning("No sector data available for visualization")
    
    # Event prediction section
    st.subheader("Predictive Analytics & Event Forecasting")
    
    # Process news data for predictions
    company_events = defaultdict(list)
    current_date = datetime.datetime.now()
    
    for item in news_data:
        try:
            pub_date = datetime.datetime.strptime(
                item['publishedAt'].replace('Z', ''),
                "%Y-%m-%dT%H:%M:%S.%f"
            )
        except:
            continue
        
        if not item.get('linkedScrips'):
            continue
            
        event_type = extract_event_type(item['headline'])
        sentiment = compute_sentiment_score(item['headline'])
        
        for company in item['linkedScrips']:
            symbol = company['symbol']
            exchange = company['exchange']
            company_events[(symbol, exchange)].append((event_type, pub_date, item['headline'], sentiment))
    
    # Create predictions
    predictions = []
    for (symbol, exchange), events in company_events.items():
        recent_events = [
            (event_type, date, headline, sentiment) 
            for event_type, date, headline, sentiment in events
            if (current_date - date).days <= 30
        ]
        
        if not recent_events:
            continue
            
        event_counts = Counter(event_type for event_type, _, _, _ in recent_events)
        if not event_counts:
            continue
            
        most_common = event_counts.most_common(1)[0][0]
        confidence = min(100, event_counts[most_common] * 20)
        
        # Skip low confidence predictions
        if confidence < min_confidence:
            continue
            
        price, pct_change, trend, history = get_stock_data(symbol, exchange)
        sentiment_avg = np.mean([sent for _, _, _, sent in recent_events]) if recent_events else 0
        
        # Get sector information
        sector = get_sector(symbol, exchange)
        
        # Calculate predictive metrics
        impact = analyze_stock_news_correlation(recent_events, history) if history is not None else 0
        clusters = detect_event_clusters(events)
        volume_spikes = detect_volume_spikes(events, history) if history is not None else 0
        reg_risk = regulatory_risk_score(events)
        trend_score = calculate_trend_score(history, sentiment_avg)
        event_duration = EVENT_DURATION.get(most_common, "unknown")
        supply_chain_impact = analyze_supply_chain_impact(sector, events, news_data)
        
        predictions.append({
            "symbol": symbol,
            "exchange": exchange,
            "sector": sector,
            "predicted_event": most_common,
            "event_duration": event_duration,
            "confidence": confidence,
            "recent_occurrences": event_counts[most_common],
            "event_clusters": clusters,
            "volume_spikes": volume_spikes,
            "current_price": price,
            "price_change_pct": pct_change,
            "price_trend": trend,
            "sentiment": sentiment_avg,
            "news_impact_pct": impact,
            "regulatory_risk": reg_risk,
            "trend_score": trend_score,
            "supply_chain_impact": supply_chain_impact
        })
    
    # Display predictions in a table
    if predictions:
        df = pd.DataFrame(predictions)
        
        # Add trend icons
        trend_icons = {
            "strong_up": "🚀",
            "moderate_up": "📈",
            "neutral": "➖",
            "moderate_down": "📉",
            "strong_down": "💥"
        }
        df['trend_icon'] = df['price_trend'].map(trend_icons)
        
        # Format columns
        df['current_price'] = df['current_price'].apply(lambda x: f"₹{x:.2f}" if x and not pd.isna(x) else "N/A")
        df['price_change_pct'] = df['price_change_pct'].apply(lambda x: f"{x:.2f}%" if x and not pd.isna(x) else "N/A")
        df['news_impact_pct'] = df['news_impact_pct'].apply(lambda x: f"{x:.2f}%" if x and not pd.isna(x) else "N/A")
        df['sentiment'] = df['sentiment'].apply(lambda x: f"{x:.1f}" if not pd.isna(x) else "N/A")
        df['supply_chain_impact'] = df['supply_chain_impact'].apply(lambda x: f"{x}%" if not pd.isna(x) else "N/A")
        
        # Color columns based on values
        def color_confidence(val):
            try:
                val = float(val)
                color = 'green' if val > 70 else 'orange' if val > 40 else 'red'
                return f'color: {color}; font-weight: bold'
            except:
                return ''
        
        def color_risk(val):
            try:
                val = float(val)
                color = 'green' if val < 30 else 'orange' if val < 70 else 'red'
                return f'color: {color}; font-weight: bold'
            except:
                return ''
        
        def color_trend(val):
            try:
                val = float(val)
                if val > 70: return 'background-color: lightgreen'
                elif val > 50: return 'background-color: lightyellow'
                else: return 'background-color: #ffcccb'
            except:
                return ''
        
        # Create styled dataframe
        styled_df = df[['symbol', 'exchange', 'sector', 'predicted_event', 'event_duration',
                         'confidence', 'recent_occurrences', 'event_clusters', 'volume_spikes',
                         'current_price', 'price_change_pct', 'trend_icon', 'sentiment',
                         'news_impact_pct', 'regulatory_risk', 'trend_score', 'supply_chain_impact'
                        ]].rename(columns={
                            'symbol': 'Symbol',
                            'exchange': 'Exchange',
                            'sector': 'Sector',
                            'predicted_event': 'Predicted Event',
                            'event_duration': 'Duration',
                            'confidence': 'Confidence',
                            'recent_occurrences': 'Events',
                            'event_clusters': 'Clusters',
                            'volume_spikes': 'Vol Spikes',
                            'current_price': 'Price',
                            'price_change_pct': '3M Chg',
                            'trend_icon': 'Trend',
                            'sentiment': 'Sentiment',
                            'news_impact_pct': 'News Impact',
                            'regulatory_risk': 'Reg Risk',
                            'trend_score': 'Trend Score',
                            'supply_chain_impact': 'Chain Impact'
                        }).style \
                         .applymap(color_confidence, subset=['Confidence']) \
                         .applymap(color_risk, subset=['Reg Risk']) \
                         .applymap(color_trend, subset=['Trend Score'])
        
        # Display table
        st.dataframe(styled_df, height=500, use_container_width=True)
        
        # Regulatory risk gauge
        st.subheader("Market Regulatory Risk Assessment")
        try:
            avg_reg_risk = df['regulatory_risk'].mean()
            if not np.isnan(avg_reg_risk):
                col1, col2 = st.columns([1, 2])
                with col1:
                    st.plotly_chart(generate_regulatory_gauge(avg_reg_risk), use_container_width=True)
                with col2:
                    st.markdown(f"""
                    **Regulatory Risk Interpretation:**
                    - **<30% (Low):** Minimal regulatory concerns
                    - **30-70% (Medium):** Moderate regulatory exposure
                    - **>70% (High):** Significant regulatory challenges
                    
                    Current market average: **{avg_reg_risk:.1f}%**
                    """)
            else:
                st.warning("No regulatory risk data available")
        except:
            st.warning("Could not calculate regulatory risk")
    else:
        st.warning("No predictions available based on current news data")
    
    # Detailed news section
    st.subheader("Latest Market News")
    
    # Filter options
    col1, col2 = st.columns(2)
    with col1:
        company_options = sorted(list(set(
            scrip['symbol'] for item in news_data 
            if 'linkedScrips' in item 
            for scrip in item['linkedScrips']
        )))
        selected_company = st.selectbox("Filter by Company", 
                                        options=['All'] + company_options)
    with col2:
        selected_event = st.selectbox("Filter by Event Type", 
                                     options=['All'] + [et[0] for et in EVENT_TYPES])
    
    # Display news cards
    displayed_count = 0
    for i, item in enumerate(news_data):
        # Apply filters
        if selected_company != 'All':
            if 'linkedScrips' not in item or not any(scrip['symbol'] == selected_company for scrip in item['linkedScrips']):
                continue
        
        event_type = extract_event_type(item['headline'])
        if selected_event != 'All' and event_type != selected_event:
            continue
        
        # Create card
        with st.expander(f"{item['headline']}", expanded=(displayed_count==0)):
            col1, col2 = st.columns([1, 3])
            
            with col1:
                if 'thumbnailImage' in item and item['thumbnailImage'] and 'url' in item['thumbnailImage']:
                    st.image(item['thumbnailImage']['url'], width=200)
            
            with col2:
                # Format date
                try:
                    pub_date = datetime.datetime.strptime(
                        item['publishedAt'].replace('Z', ''),
                        "%Y-%m-%dT%H:%M:%S.%f"
                    )
                    st.caption(f"Published: {pub_date.strftime('%b %d, %Y %H:%M')}")
                except:
                    st.caption("Published: N/A")
                
                # Show event type with color coding
                event_color = {
                    'acquisition': 'blue',
                    'partnership': 'green',
                    'agreement': 'orange',
                    'investment': 'purple',
                    'launch': 'red',
                    'expansion': 'teal',
                    'award': 'gold',
                    'leadership': 'pink',
                    'financial': 'brown',
                    'regulatory': 'gray'
                }.get(event_type, 'black')
                
                st.markdown(f"**Event Type:** <span style='color:{event_color}; font-weight:bold'>{event_type.title()}</span>", 
                            unsafe_allow_html=True)
                
                # Sentiment indicator
                sentiment = compute_sentiment_score(item['headline'])
                sentiment_icon = "😊" if sentiment > 0 else "😐" if sentiment == 0 else "😞"
                sentiment_text = "Positive" if sentiment > 0 else "Neutral" if sentiment == 0 else "Negative"
                sentiment_color = "green" if sentiment > 0 else "gray" if sentiment == 0 else "red"
                st.markdown(f"**Sentiment:** <span style='color:{sentiment_color}; font-weight:bold'>{sentiment_text} {sentiment_icon}</span>", 
                            unsafe_allow_html=True)
                
                if 'summary' in item:
                    st.write(item['summary'])
                else:
                    st.write("No summary available")
                
                # Show linked companies
                if 'linkedScrips' in item and item['linkedScrips']:
                    companies = ", ".join(scrip['symbol'] for scrip in item['linkedScrips'])
                    st.markdown(f"**Related Companies:** {companies}")
                
                if 'contentUrl' in item:
                    st.markdown(f"[Read full article]({item['contentUrl']})")
                else:
                    st.write("No article link available")
        
        displayed_count += 1
        if displayed_count >= 10:  # Limit to 10 news items
            break

if __name__ == "__main__":
    main()
