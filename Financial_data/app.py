import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

# Konfiguracja strony
st.set_page_config(
    page_title="Dashboard Finansowy",
    page_icon="📈",
    layout="wide"
)

# Stylizacja
st.markdown("""
<style>
.main {
    background-color: #f5f5f5;
}
.stApp {
    max-width: 1200px;
    margin: 0 auto;
}
h1, h2, h3 {
    color: #1E3A8A;
}
.metric-card {
    background-color: #f9f9f9;
    border-radius: 10px;
    padding: 20px;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
}
.plot-container {
    background-color: #f9f9f9;
    border-radius: 10px;
    padding: 20px;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    margin-bottom: 20px;
}
.stTabs [data-baseweb="tab-list"] {
    gap: 24px;
}
.stTabs [data-baseweb="tab"] {
    height: 50px;
    white-space: pre-wrap;
    background-color: #f9f9f9;
    border-radius: 4px 4px 0px 0px;
    gap: 1px;
    padding-top: 10px;
    padding-bottom: 10px;
}
.stTabs [aria-selected="true"] {
    background-color: #1E3A8A;
    color: #f9f9f9;
}
.metric-card {
    background-color: #f9f9f9;
    border-radius: 10px;
    padding: 20px;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    color: #333333;  /* <- dodaj to */
}
</style>
""", unsafe_allow_html=True)

# Funkcja do pobierania danych o akcjach
@st.cache_data(ttl=3600)
def load_stock_data(ticker, period, interval):
    stock = yf.Ticker(ticker)
    hist = stock.history(period=period, interval=interval)
    if hist.empty:
        st.error(f"Nie znaleziono danych dla {ticker}")
        return None
    return hist

# Funkcja do pobierania informacji o firmie
@st.cache_data(ttl=86400)
def get_company_info(ticker):
    stock = yf.Ticker(ticker)
    info = stock.info
    return info

# Funkcja do pobierania rekomendacji analityków
@st.cache_data(ttl=86400)
def get_recommendations(ticker):
    stock = yf.Ticker(ticker)
    try:
        recommendations = stock.recommendations
        if recommendations is not None and not recommendations.empty:
            return recommendations.iloc[-10:]
        return None
    except:
        return None

# Funkcja do obliczania wskaźników technicznych
def calculate_indicators(data):
    # Średnie kroczące
    data['MA20'] = data['Close'].rolling(window=20).mean()
    data['MA50'] = data['Close'].rolling(window=50).mean()
    data['MA200'] = data['Close'].rolling(window=200).mean()
    
    # RSI
    delta = data['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    data['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD
    data['EMA12'] = data['Close'].ewm(span=12, adjust=False).mean()
    data['EMA26'] = data['Close'].ewm(span=26, adjust=False).mean()
    data['MACD'] = data['EMA12'] - data['EMA26']
    data['Signal'] = data['MACD'].ewm(span=9, adjust=False).mean()
    
    # Bolinger Bands
    data['MA20_std'] = data['Close'].rolling(window=20).std()
    data['Upper_Band'] = data['MA20'] + (data['MA20_std'] * 2)
    data['Lower_Band'] = data['MA20'] - (data['MA20_std'] * 2)
    
    return data

# Funkcja do generowania analizy i rekomendacji
def generate_analysis(data):
    last_price = data['Close'].iloc[-1]
    prev_price = data['Close'].iloc[-2]
    change = (last_price - prev_price) / prev_price * 100
    
    analysis = {}
    
    # Analiza trendu
    ma20 = data['MA20'].iloc[-1]
    ma50 = data['MA50'].iloc[-1]
    ma200 = data['MA200'].iloc[-1]
    
    if last_price > ma20 and last_price > ma50:
        analysis['trend'] = "Wzrostowy (krótki termin)"
    elif last_price < ma20 and last_price < ma50:
        analysis['trend'] = "Spadkowy (krótki termin)"
    else:
        analysis['trend'] = "Boczny (krótki termin)"
        
    if last_price > ma200:
        analysis['long_trend'] = "Wzrostowy (długi termin)"
    else:
        analysis['long_trend'] = "Spadkowy (długi termin)"
    
    # Analiza RSI
    rsi = data['RSI'].iloc[-1]
    if rsi > 70:
        analysis['rsi'] = f"Wykupienie - {rsi:.2f}"
    elif rsi < 30:
        analysis['rsi'] = f"Wyprzedanie - {rsi:.2f}"
    else:
        analysis['rsi'] = f"Neutralny - {rsi:.2f}"
    
    # Analiza MACD
    macd = data['MACD'].iloc[-1]
    signal = data['Signal'].iloc[-1]
    
    if macd > signal and data['MACD'].iloc[-2] <= data['Signal'].iloc[-2]:
        analysis['macd'] = "Sygnał kupna (MACD przebija linię sygnałową od dołu)"
    elif macd < signal and data['MACD'].iloc[-2] >= data['Signal'].iloc[-2]:
        analysis['macd'] = "Sygnał sprzedaży (MACD przebija linię sygnałową od góry)"
    elif macd > signal:
        analysis['macd'] = "Pozytywny (MACD powyżej linii sygnałowej)"
    else:
        analysis['macd'] = "Negatywny (MACD poniżej linii sygnałowej)"
    
    # Analiza Bollinger Bands
    upper = data['Upper_Band'].iloc[-1]
    lower = data['Lower_Band'].iloc[-1]
    
    if last_price > upper:
        analysis['bollinger'] = "Powyżej górnego pasma - możliwe wykupienie"
    elif last_price < lower:
        analysis['bollinger'] = "Poniżej dolnego pasma - możliwe wyprzedanie"
    else:
        analysis['bollinger'] = "W zakresie pasm - neutralny"
    
    return analysis

# Funkcja do obliczania portfela i symulacji inwestycji
def simulate_portfolio(tickers, weights, investment, start_date):
    end_date = datetime.now()
    
    # Sprawdzenie poprawności wag
    if sum(weights) != 1.0:
        st.warning("Suma wag powinna wynosić 1.0. Normalizuję wagi.")
        weights = [w/sum(weights) for w in weights]
    
    # Pobieranie danych
    portfolio_data = {}
    for ticker in tickers:
        stock = yf.Ticker(ticker)
        hist = stock.history(start=start_date, end=end_date)
        if not hist.empty:
            portfolio_data[ticker] = hist['Close']
    
    if not portfolio_data:
        return None
    
    # Tworzenie DataFrame z cenami zamknięcia
    df = pd.DataFrame(portfolio_data)
    
    # Obliczanie zwrotów dziennych
    returns = df.pct_change().dropna()
    
    # Obliczanie skumulowanych zwrotów
    cumulative_returns = (1 + returns).cumprod()
    
    # Obliczanie wartości portfela
    portfolio_value = pd.DataFrame()
    
    for i, ticker in enumerate(tickers):
        if ticker in df.columns:
            investment_per_ticker = investment * weights[i]
            shares = investment_per_ticker / df[ticker].iloc[0]
            ticker_value = df[ticker] * shares
            portfolio_value[ticker] = ticker_value
    
    portfolio_value['Total'] = portfolio_value.sum(axis=1)
    
    # Obliczanie metryk portfela
    initial_value = investment
    final_value = portfolio_value['Total'].iloc[-1]
    total_return = (final_value - initial_value) / initial_value * 100
    
    # Annualized return
    days = (end_date - datetime.strptime(start_date, '%Y-%m-%d')).days
    annual_return = ((1 + total_return/100) ** (365/days) - 1) * 100
    
    # Volatility
    daily_returns = portfolio_value['Total'].pct_change().dropna()
    volatility = daily_returns.std() * np.sqrt(252) * 100
    
    # Sharpe Ratio (assuming risk-free rate of 2%)
    risk_free = 0.02
    sharpe = (annual_return/100 - risk_free) / (volatility/100)
    
    # Maximum Drawdown
    cumulative_returns = (1 + daily_returns).cumprod()
    peak = cumulative_returns.expanding().max()
    drawdown = (cumulative_returns / peak - 1) * 100
    max_drawdown = drawdown.min()
    
    portfolio_metrics = {
        'Initial Investment': f"${investment:,.2f}",
        'Current Value': f"${final_value:,.2f}",
        'Total Return': f"{total_return:.2f}%",
        'Annualized Return': f"{annual_return:.2f}%",
        'Volatility (Annualized)': f"{volatility:.2f}%",
        'Sharpe Ratio': f"{sharpe:.2f}",
        'Maximum Drawdown': f"{max_drawdown:.2f}%"
    }
    
    return {
        'portfolio_value': portfolio_value,
        'metrics': portfolio_metrics
    }

# Główny nagłówek
st.title("📈 Dashboard Finansowy")
st.markdown("Interaktywne narzędzie do analizy danych finansowych i rynków akcji")

# Zakładki dla różnych sekcji
tabs = st.tabs(["Analiza Akcji", "Porównanie Spółek", "Analiza Portfela", "Wskaźniki Rynkowe"])

# Zakładka 1: Analiza Akcji
with tabs[0]:
    st.header("Analiza Pojedynczej Akcji")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        popular_tickers = {
    "Apple (AAPL)": "AAPL",
    "Microsoft (MSFT)": "MSFT",
    "Google (GOOGL)": "GOOGL",
    "Amazon (AMZN)": "AMZN",
    "Meta (META)": "META",
    "Tesla (TSLA)": "TSLA",
    "Nvidia (NVDA)": "NVDA",
    "Netflix (NFLX)": "NFLX",
    "Intel (INTC)": "INTC",
    "AMD (AMD)": "AMD"
}

        selected_label = st.selectbox("Wybierz spółkę", list(popular_tickers.keys()))
        ticker = popular_tickers[selected_label]
        period_options = {
            "1 dzień": "1d", 
            "5 dni": "5d", 
            "1 miesiąc": "1mo", 
            "3 miesiące": "3mo",
            "6 miesięcy": "6mo", 
            "1 rok": "1y", 
            "2 lata": "2y", 
            "5 lat": "5y"
        }
        period = st.selectbox("Okres", list(period_options.keys()))
        
        interval_options = {
            "5 minut": "5m", 
            "15 minut": "15m", 
            "30 minut": "30m", 
            "1 godzina": "1h",
            "1 dzień": "1d", 
            "1 tydzień": "1wk", 
            "1 miesiąc": "1mo"
        }
        interval = st.selectbox("Interwał", list(interval_options.keys()))
        
        analyze_btn = st.button("Analizuj")
    
    if analyze_btn or ticker:
        with st.spinner(f"Pobieranie danych dla {ticker}..."):
            # Pobieranie danych
            data = load_stock_data(ticker, period_options[period], interval_options[interval])
            
            if data is not None:
                # Pobieranie informacji o firmie
                company_info = get_company_info(ticker)
                
                # Obliczanie wskaźników
                data = calculate_indicators(data)
                
                # Generowanie analizy
                analysis = generate_analysis(data)
                
                # Pobieranie rekomendacji
                recommendations = get_recommendations(ticker)
                
                # Wyświetlanie informacji o firmie
                with col1:
                    st.markdown(f"<h3>{company_info.get('shortName', ticker)}</h3>", unsafe_allow_html=True)
                    st.text(company_info.get('sector', 'Brak sektora') + " | " + company_info.get('industry', 'Brak branży'))
                    
                    # Wyświetlanie metryki
                    current_price = data['Close'].iloc[-1]
                    previous_price = data['Close'].iloc[-2]
                    price_change = current_price - previous_price
                    price_change_pct = (price_change / previous_price) * 100
                    
                    color = "green" if price_change >= 0 else "red"
                    arrow = "↑" if price_change >= 0 else "↓"
                    
                    st.markdown(f"""
                    <div class="metric-card">
                        <h4>Aktualna cena</h4>
                        <h2>${current_price:.2f} <span style="color:{color};">{arrow} {abs(price_change):.2f} ({abs(price_change_pct):.2f}%)</span></h2>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Wyświetlanie analizy
                    st.markdown("<h4>Analiza techniczna</h4>", unsafe_allow_html=True)
                    st.markdown(f"""
                    <div class="metric-card">
                        <p><strong>Trend: </strong>{analysis['trend']}</p>
                        <p><strong>Trend długoterminowy: </strong>{analysis['long_trend']}</p>
                        <p><strong>RSI: </strong>{analysis['rsi']}</p>
                        <p><strong>MACD: </strong>{analysis['macd']}</p>
                        <p><strong>Bollinger Bands: </strong>{analysis['bollinger']}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Wyświetlanie wykresów
                with col2:
                    # Wykres świecowy
                    fig = go.Figure()
                    fig.add_trace(go.Candlestick(
                        x=data.index,
                        open=data['Open'],
                        high=data['High'],
                        low=data['Low'],
                        close=data['Close'],
                        name='OHLC'
                    ))
                    
                    # Dodawanie średnich kroczących
                    fig.add_trace(go.Scatter(x=data.index, y=data['MA20'], name='MA20', line=dict(color='blue', width=1)))
                    fig.add_trace(go.Scatter(x=data.index, y=data['MA50'], name='MA50', line=dict(color='orange', width=1)))
                    fig.add_trace(go.Scatter(x=data.index, y=data['MA200'], name='MA200', line=dict(color='red', width=1)))
                    
                    # Dodawanie Bollinger Bands
                    fig.add_trace(go.Scatter(x=data.index, y=data['Upper_Band'], name='Upper Band', line=dict(color='rgba(0,100,0,0.3)', width=1)))
                    fig.add_trace(go.Scatter(x=data.index, y=data['Lower_Band'], name='Lower Band', line=dict(color='rgba(0,100,0,0.3)', width=1), fill='tonexty'))
                    
                    fig.update_layout(
                        title=f"{ticker} - Wykres świecowy z indykatorami",
                        xaxis_title="Data",
                        yaxis_title="Cena ($)",
                        height=500,
                        margin=dict(l=0, r=0, t=40, b=0)
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Wykres wolumenu
                    fig_volume = go.Figure()
                    fig_volume.add_trace(go.Bar(
                        x=data.index,
                        y=data['Volume'],
                        name='Wolumen',
                        marker=dict(color='rgba(0, 0, 139, 0.5)')
                    ))
                    
                    fig_volume.update_layout(
                        title="Wolumen",
                        xaxis_title="Data",
                        yaxis_title="Wolumen",
                        height=200,
                        margin=dict(l=0, r=0, t=40, b=0)
                    )
                    
                    st.plotly_chart(fig_volume, use_container_width=True)
                    
                    # Wykres RSI
                    fig_rsi = go.Figure()
                    fig_rsi.add_trace(go.Scatter(
                        x=data.index,
                        y=data['RSI'],
                        name='RSI',
                        line=dict(color='purple', width=2)
                    ))
                    
                    # Dodawanie linii na poziomach 30 i 70
                    fig_rsi.add_shape(
                        type='line',
                        x0=data.index[0],
                        y0=30,
                        x1=data.index[-1],
                        y1=30,
                        line=dict(color='green', width=2, dash='dash')
                    )
                    
                    fig_rsi.add_shape(
                        type='line',
                        x0=data.index[0],
                        y0=70,
                        x1=data.index[-1],
                        y1=70,
                        line=dict(color='red', width=2, dash='dash')
                    )
                    
                    fig_rsi.update_layout(
                        title="Wskaźnik RSI",
                        xaxis_title="Data",
                        yaxis_title="RSI",
                        height=200,
                        margin=dict(l=0, r=0, t=40, b=0)
                    )
                    
                    st.plotly_chart(fig_rsi, use_container_width=True)
                    
                    # Wykres MACD
                    fig_macd = go.Figure()
                    fig_macd.add_trace(go.Scatter(
                        x=data.index,
                        y=data['MACD'],
                        name='MACD',
                        line=dict(color='blue', width=2)
                    ))
                    
                    fig_macd.add_trace(go.Scatter(
                        x=data.index,
                        y=data['Signal'],
                        name='Signal',
                        line=dict(color='red', width=2)
                    ))
                    
                    # Dodawanie histogramu różnicy
                    fig_macd.add_trace(go.Bar(
                        x=data.index,
                        y=data['MACD'] - data['Signal'],
                        name='Histogram',
                        marker=dict(color='rgba(0, 0, 139, 0.5)')
                    ))
                    
                    fig_macd.update_layout(
                        title="Wskaźnik MACD",
                        xaxis_title="Data",
                        yaxis_title="MACD",
                        height=200,
                        margin=dict(l=0, r=0, t=40, b=0)
                    )
                    
                    st.plotly_chart(fig_macd, use_container_width=True)
                    
                    # Wyświetlanie rekomendacji
                    if recommendations is not None and not recommendations.empty:
                        st.subheader("Rekomendacje analityków")
                        
                        # Przygotowanie danych do wykresu
                        recommendations_melted = recommendations.melt(
    id_vars='period',
    value_vars=['strongBuy', 'buy', 'hold', 'sell', 'strongSell'],
    var_name='Recommendation Type',
    value_name='Count'
)

# Sumuj po typie rekomendacji
                        recommendations_count = recommendations_melted.groupby('Recommendation Type')['Count'].sum()

                        st.bar_chart(recommendations_count)
                        
                        fig_recommendations = go.Figure()
                        fig_recommendations.add_trace(go.Bar(
                            x=recommendations_count.index,
                            y=recommendations_count.values,
                            marker_color=['green' if x in ['Buy', 'Outperform', 'Strong Buy'] 
                                          else 'red' if x in ['Sell', 'Underperform', 'Strong Sell'] 
                                          else 'gray' for x in recommendations_count.index]
                        ))
                        
                        fig_recommendations.update_layout(
                            title="Rozkład rekomendacji",
                            xaxis_title="Rekomendacja",
                            yaxis_title="Liczba",
                            height=300,
                            margin=dict(l=0, r=0, t=40, b=0)
                        )
                        
                        st.plotly_chart(fig_recommendations, use_container_width=True)
                        
                        # Tabela rekomendacji
                        st.dataframe(recommendations[['strongBuy', 'buy', 'hold', 'sell', 'strongSell']].style.background_gradient(cmap='Blues'))


# Zakładka 2: Porównanie Spółek
with tabs[1]:
    st.header("Porównanie Spółek")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        ticker_list = st.text_input("Symbole akcji (oddzielone przecinkami)", "AAPL,MSFT,GOOGL")
        tickers = [ticker.strip() for ticker in ticker_list.split(",")]
        
        period_options = {
            "1 miesiąc": "1mo", 
            "3 miesiące": "3mo",
            "6 miesięcy": "6mo", 
            "1 rok": "1y", 
            "2 lata": "2y", 
            "5 lat": "5y"
        }
        compare_period = st.selectbox("Okres porównania", list(period_options.keys()))
        
        metric_options = ["Cena", "Procentowa zmiana", "Wolumen"]
        compare_metric = st.selectbox("Metryka do porównania", metric_options)
        
        compare_btn = st.button("Porównaj")
    
    if compare_btn or ticker_list:
        with st.spinner("Pobieranie danych do porównania..."):
            all_data = {}
            company_infos = {}
            
            for ticker in tickers:
                data = load_stock_data(ticker, period_options[compare_period], "1d")
                if data is not None:
                    all_data[ticker] = data
                    company_infos[ticker] = get_company_info(ticker)
            
            if all_data:
                # Przygotowanie danych do porównania
                compare_data = pd.DataFrame()
                
                if compare_metric == "Cena":
                    for ticker in all_data:
                        compare_data[ticker] = all_data[ticker]['Close']
                elif compare_metric == "Procentowa zmiana":
                    for ticker in all_data:
                        first_value = all_data[ticker]['Close'].iloc[0]
                        compare_data[ticker] = (all_data[ticker]['Close'] / first_value - 1) * 100
                elif compare_metric == "Wolumen":
                    for ticker in all_data:
                        compare_data[ticker] = all_data[ticker]['Volume']
                
                # Wyświetlanie porównania
                with col2:
                    # Wykres porównawczy
                    fig = go.Figure()
                    
                    for ticker in compare_data.columns:
                        fig.add_trace(go.Scatter(
                            x=compare_data.index,
                            y=compare_data[ticker],
                            name=ticker,
                            mode='lines'
                        ))
                    
                    y_title = "Cena ($)" if compare_metric == "Cena" else "Zmiana (%)" if compare_metric == "Procentowa zmiana" else "Wolumen"
                    
                    fig.update_layout(
                        title=f"Porównanie {compare_metric.lower()} spółek",
                        xaxis_title="Data",
                        yaxis_title=y_title,
                        height=500,
                        margin=dict(l=0, r=0, t=40, b=0)
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Tabela porównawcza
                    comparison_table = {}
                    
                    for ticker in all_data:
                        data = all_data[ticker]
                        info = company_infos.get(ticker, {})
                        
                        current_price = data['Close'].iloc[-1]
                        start_price = data['Close'].iloc[0]
                        price_change = (current_price - start_price) / start_price * 100
                        
                        comparison_table[ticker] = {
                            'Nazwa': info.get('shortName', ticker),
                            'Sektor': info.get('sector', 'N/A'),
                            'Aktualna cena': f"${current_price:.2f}",
                            f'Zmiana ({compare_period})': f"{price_change:.2f}%",
                            'Kapitalizacja rynkowa': f"${info.get('marketCap', 0)/1e9:.2f}B",
                            'P/E': info.get('trailingPE', 'N/A'),
                            'Dywidenda (%)': info.get('dividendYield', 'N/A') * 100 if info.get('dividendYield') else 'N/A'
                        }
                    
                    comparison_df = pd.DataFrame.from_dict(comparison_table, orient='index')
                    st.dataframe(comparison_df.style.highlight_max(axis=0, subset=[f'Zmiana ({compare_period})']))
                
                # Korelacja
                if len(compare_data.columns) > 1:
                    st.subheader("Macierz korelacji")
                    
                    # Obliczanie korelacji
                    correlation = compare_data.corr()
                    
                    # Rysowanie heatmapy korelacji
                    fig_corr = px.imshow(
                        correlation,
                        text_auto=True,
                        color_continuous_scale='RdBu_r',
                        zmin=-1, zmax=1,
                        aspect="auto"
                    )
                    
                    fig_corr.update_layout(
                        title="Korelacja między spółkami",
                        height=400,
                        margin=dict(l=0, r=0, t=40, b=0)
                    )
                    
                    st.plotly_chart(fig_corr, use_container_width=True)

# Zakładka 3: Analiza Portfela
with tabs[2]:
    st.header("Symulacja i Analiza Portfela")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        portfolio_tickers = st.text_input("Symbole akcji w portfelu (oddzielone przecinkami)", "AAPL,MSFT,GOOGL,AMZN")
        portfolio_ticker_list = [ticker.strip() for ticker in portfolio_tickers.split(",")]
        
        # Dynamiczne inputy dla wag
        weights = []
        weights_col1, weights_col2 = st.columns(2)
        
        for i, ticker in enumerate(portfolio_ticker_list):
            if i % 2 == 0:
                weight = weights_col1.number_input(f"Waga {ticker}", min_value=0.0, max_value=1.0, value=1.0/len(portfolio_ticker_list), step=0.05)
            else:
                weight = weights_col2.number_input(f"Waga {ticker}", min_value=0.0, max_value=1.0, value=1.0/len(portfolio_ticker_list), step=0.05)
            weights.append(weight)
        
        investment = st.number_input("Kwota inwestycji ($)", min_value=1000, value=10000, step=1000)
        start_date = st.date_input("Data początkowa", value=datetime.now() - relativedelta(years=1)).strftime('%Y-%m-%d')
        
        simulate_btn = st.button("Symuluj portfel")
    
    if simulate_btn or portfolio_tickers:
        with st.spinner("Symulacja portfela..."):
            portfolio_results = simulate_portfolio(portfolio_ticker_list, weights, investment, start_date)
            
            if portfolio_results:
                portfolio_value = portfolio_results['portfolio_value']
                metrics = portfolio_results['metrics']
                
                # Wyświetlanie wyników
                with col2:
                    # Wykres wartości portfela
                    # Wykres wartości portfela
                    fig = go.Figure()
                    
                    # Dodawanie wartości poszczególnych akcji
                    for ticker in portfolio_ticker_list:
                        if ticker in portfolio_value.columns and ticker != 'Total':
                            fig.add_trace(go.Scatter(
                                x=portfolio_value.index,
                                y=portfolio_value[ticker],
                                name=ticker,
                                stackgroup='portfolio'
                            ))
                    
                    fig.update_layout(
                        title="Struktura wartości portfela w czasie",
                        xaxis_title="Data",
                        yaxis_title="Wartość ($)",
                        height=400,
                        margin=dict(l=0, r=0, t=40, b=0)
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Wykres całkowitej wartości portfela
                    fig_total = go.Figure()
                    fig_total.add_trace(go.Scatter(
                        x=portfolio_value.index,
                        y=portfolio_value['Total'],
                        name='Wartość całkowita',
                        line=dict(width=3, color='rgb(0, 100, 80)')
                    ))
                    
                    # Dodawanie linii początkowej inwestycji
                    fig_total.add_shape(
                        type='line',
                        x0=portfolio_value.index[0],
                        y0=investment,
                        x1=portfolio_value.index[-1],
                        y1=investment,
                        line=dict(color='red', width=2, dash='dash')
                    )
                    
                    fig_total.update_layout(
                        title="Zmiana wartości całkowitej portfela",
                        xaxis_title="Data",
                        yaxis_title="Wartość ($)",
                        height=300,
                        margin=dict(l=0, r=0, t=40, b=0)
                    )
                    
                    st.plotly_chart(fig_total, use_container_width=True)
                    
                    # Wyświetlanie metryk portfela
                    st.subheader("Metryki portfela")
                    
                    metrics_col1, metrics_col2 = st.columns(2)
                    
                    metrics_col1.metric("Początkowa inwestycja", metrics['Initial Investment'])
                    metrics_col2.metric("Aktualna wartość", metrics['Current Value'])
                    
                    metrics_col1.metric("Całkowity zwrot", metrics['Total Return'])
                    metrics_col2.metric("Roczny zwrot", metrics['Annualized Return'])
                    
                    metrics_col1.metric("Zmienność (roczna)", metrics['Volatility (Annualized)'])
                    metrics_col2.metric("Wskaźnik Sharpe'a", metrics['Sharpe Ratio'])
                    
                    metrics_col1.metric("Maksymalne obsunięcie", metrics['Maximum Drawdown'])
                    
                    # Wykres struktury portfela (pie chart)
                    current_values = {}
                    for ticker in portfolio_ticker_list:
                        if ticker in portfolio_value.columns and ticker != 'Total':
                            current_values[ticker] = portfolio_value[ticker].iloc[-1]
                    
                    fig_pie = go.Figure(data=[go.Pie(
                        labels=list(current_values.keys()),
                        values=list(current_values.values()),
                        hole=.3
                    )])
                    
                    fig_pie.update_layout(
                        title="Aktualna struktura portfela",
                        height=400,
                        margin=dict(l=0, r=0, t=40, b=0)
                    )
                    
                    st.plotly_chart(fig_pie, use_container_width=True)

# Zakładka 4: Wskaźniki Rynkowe
with tabs[3]:
    st.header("Wskaźniki Rynkowe")
    
    # Indeksy do śledzenia
    indices = {
        "S&P 500": "^GSPC",
        "Nasdaq": "^IXIC",
        "Dow Jones": "^DJI",
        "Russell 2000": "^RUT",
        "FTSE 100": "^FTSE",
        "DAX": "^GDAXI",
        "Nikkei 225": "^N225"
    }
    
    # Wybór indeksów
    selected_indices = st.multiselect("Wybierz indeksy do wyświetlenia", list(indices.keys()), default=["S&P 500", "Nasdaq", "Dow Jones"])
    
    period_options = {
        "1 miesiąc": "1mo", 
        "3 miesiące": "3mo",
        "6 miesięcy": "6mo", 
        "1 rok": "1y", 
        "2 lata": "2y", 
        "5 lat": "5y"
    }
    index_period = st.selectbox("Okres dla indeksów", list(period_options.keys()), key="indices_period")
    
    track_btn = st.button("Śledź indeksy")
    
    if track_btn or selected_indices:
        with st.spinner("Pobieranie danych o indeksach..."):
            # Pobieranie danych indeksów
            indices_data = {}
            
            for index_name in selected_indices:
                ticker = indices[index_name]
                data = load_stock_data(ticker, period_options[index_period], "1d")
                if data is not None:
                    indices_data[index_name] = data
            
            if indices_data:
                # Przygotowanie danych do wykresów
                indices_close = pd.DataFrame()
                indices_change = pd.DataFrame()
                
                for index_name in indices_data:
                    indices_close[index_name] = indices_data[index_name]['Close']
                    first_value = indices_data[index_name]['Close'].iloc[0]
                    indices_change[index_name] = (indices_data[index_name]['Close'] / first_value - 1) * 100
                
                # Wykres wartości indeksów
                fig_indices = go.Figure()
                
                for index_name in indices_close.columns:
                    fig_indices.add_trace(go.Scatter(
                        x=indices_close.index,
                        y=indices_close[index_name],
                        name=index_name
                    ))
                
                fig_indices.update_layout(
                    title="Wartości indeksów",
                    xaxis_title="Data",
                    yaxis_title="Wartość",
                    height=400,
                    margin=dict(l=0, r=0, t=40, b=0),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
                )
                
                st.plotly_chart(fig_indices, use_container_width=True)
                
                # Wykres procentowej zmiany indeksów
                fig_change = go.Figure()
                
                for index_name in indices_change.columns:
                    fig_change.add_trace(go.Scatter(
                        x=indices_change.index,
                        y=indices_change[index_name],
                        name=index_name
                    ))
                
                fig_change.update_layout(
                    title="Procentowa zmiana indeksów",
                    xaxis_title="Data",
                    yaxis_title="Zmiana (%)",
                    height=400,
                    margin=dict(l=0, r=0, t=40, b=0),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
                )
                
                st.plotly_chart(fig_change, use_container_width=True)
                
                # Tabela z aktualnymi wartościami i zmianami
                indices_table = pd.DataFrame(columns=["Indeks", "Aktualna wartość", "Dzienna zmiana", "Dzienna zmiana (%)", f"Zmiana w okresie {index_period}", f"Zmiana w okresie {index_period} (%)"])
                
                for i, index_name in enumerate(indices_data):
                    data = indices_data[index_name]
                    current_value = data['Close'].iloc[-1]
                    previous_value = data['Close'].iloc[-2]
                    start_value = data['Close'].iloc[0]
                    
                    daily_change = current_value - previous_value
                    daily_change_pct = (daily_change / previous_value) * 100
                    
                    period_change = current_value - start_value
                    period_change_pct = (period_change / start_value) * 100
                    
                    indices_table.loc[i] = [
                        index_name, 
                        f"{current_value:.2f}", 
                        f"{daily_change:.2f}", 
                        f"{daily_change_pct:.2f}%",
                        f"{period_change:.2f}",
                        f"{period_change_pct:.2f}%"
                    ]
                
                st.dataframe(indices_table.style.apply(lambda x: ['color: green' if float(val.replace('%', '')) > 0 else 'color: red' if float(val.replace('%', '')) < 0 else '' for val in x], 
                                                 subset=['Dzienna zmiana (%)', f'Zmiana w okresie {index_period} (%)']), use_container_width=True)
    
    # Sekcja z danymi ekonomicznymi
    st.subheader("Wskaźniki ekonomiczne")
    st.info("Uwaga: Dane wskaźników ekonomicznych są symulowane na potrzeby demonstracji. W rzeczywistej aplikacji można użyć API dostawców danych ekonomicznych.")
    
    # Symulowane dane ekonomiczne
    economic_data = {
        "Inflacja (CPI)": {"value": "3.5%", "previous": "3.7%", "trend": "↓"},
        "Stopa bezrobocia": {"value": "4.1%", "previous": "4.2%", "trend": "↓"},
        "PKB (kwartalny wzrost)": {"value": "2.3%", "previous": "2.1%", "trend": "↑"},
        "Stopa procentowa FED": {"value": "4.75%", "previous": "4.75%", "trend": "→"},
        "Rentowność 10-letnich obligacji": {"value": "3.85%", "previous": "3.91%", "trend": "↓"}
    }
    
    # Wyświetlanie danych ekonomicznych
    col1, col2, col3 = st.columns(3)
    
    for i, (indicator, data) in enumerate(economic_data.items()):
        col = [col1, col2, col3][i % 3]
        
        color = "green" if data["trend"] == "↑" else "red" if data["trend"] == "↓" else "gray"
        
        col.markdown(f"""
        <div class="metric-card" style="height: 130px;">
            <h4>{indicator}</h4>
            <h2>{data['value']} <span style="color:{color};">{data['trend']}</span></h2>
            <p>Poprzednio: {data['previous']}</p>
        </div>
        """, unsafe_allow_html=True)

# Dodatkowe informacje
st.markdown("""
---
### O dashboardzie
Ten interaktywny dashboard finansowy został stworzony przy użyciu Python i Streamlit. Umożliwia on:
- Analizę pojedynczych akcji z wykorzystaniem wskaźników technicznych
- Porównanie wielu spółek
- Symulację i analizę portfela inwestycyjnego
- Śledzenie głównych indeksów giełdowych i wskaźników ekonomicznych

**Uwaga**: Dashboard wykorzystuje dane z Yahoo Finance (yfinance) do pobierania informacji o akcjach i indeksach. Wskaźniki ekonomiczne są symulowane na potrzeby demonstracji.

**Autor**: Maciej Zabdyr - https://github.com/maciegg5

---
""")

st.markdown("""
    <style>
        /* Styl dla aktywnej zakładki */
        .stTabs [data-baseweb="tab"] {
            background-color: #f0f2f6;
            color: black;
            border-radius: 5px 5px 0 0;
            padding: 0.5rem 1rem;
        }

        .stTabs [data-baseweb="tab"]:hover {
            background-color: #e0e0e0;
            color: black;
        }

        .stTabs [aria-selected="true"] {
            background-color: #2c6ebe !important;
            color: white !important;
        }
    </style>
""", unsafe_allow_html=True)
