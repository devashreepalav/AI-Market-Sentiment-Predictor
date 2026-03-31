import streamlit as st
import yfinance as yf
import joblib
import pandas as pd
from newsapi import NewsApiClient
from textblob import TextBlob

# --- 1. SETTINGS & API CONFIG ---
# Get your key at https://newsapi.org/
NEWS_API_KEY = 'e008faf9743c4c33b332718686b32bad' 
newsapi = NewsApiClient(api_key=e008faf9743c4c33b332718686b32bad)

st.set_page_config(page_title="AI Market Vibe", layout="wide", page_icon="📈")

# --- 2. LOAD TRAINED MODEL ---
@st.cache_resource
def load_ai_model():
    try:
        model = joblib.load('stock_predictor_model.pkl')
        features = joblib.load('model_features.pkl')
        return model, features
    except Exception as e:
        st.error(f"Error loading model files: {e}")
        return None, None

model, model_features = load_ai_model()

# --- 3. SIDEBAR ---
st.sidebar.header("🕹️ Control Panel")
ticker = st.sidebar.text_input("Enter Stock Ticker", "AAPL").upper()
days_to_show = st.sidebar.slider("Days of History", 30, 365, 100)

st.title(f"🚀 AI Market Insight: {ticker}")

# --- 4. FETCH & FIX DATA ---
try:
    # Fetching data with auto_adjust=True to simplify the OHLC structure
    data = yf.download(ticker, period=f"{days_to_show + 100}d", interval="1d", auto_adjust=True)

    if not data.empty:
        # 🔥 THE BULLETPROOF FIX:
        # No matter what, we only keep the first level of names (Close, Volume, etc.)
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        
        # Reset column names to standard strings to remove any hidden ticker names
        data.columns = [str(col) for col in data.columns]

        # Force calculate the EXACT features your model is looking for
        # We use ['Close'] specifically to avoid any multi-column confusion
        data['Returns'] = data['Close'].pct_change()
        data['MA10'] = data['Close'].rolling(window=10).mean()
        data['MA50'] = data['Close'].rolling(window=50).mean()
        
        # Drop the first 50 empty rows
        data = data.dropna()
        
        if not data.empty:
            latest_row = data.tail(1)

            # --- 5. PREDICTION UI ---
            if model is not None:
                # We create a brand new DataFrame with ONLY the 5 features the model needs
                # This bypasses any "Index" or "Multi-Index" errors entirely
                input_data = pd.DataFrame([{
                    'Close': float(latest_row['Close'].iloc[0]),
                    'Volume': float(latest_row['Volume'].iloc[0]),
                    'MA10': float(latest_row['MA10'].iloc[0]),
                    'MA50': float(latest_row['MA50'].iloc[0]),
                    'Returns': float(latest_row['Returns'].iloc[0])
                }])
                
                prediction = model.predict(input_data)[0]
                confidence = model.predict_proba(input_data).max() * 100
                
                col_pred, col_conf = st.columns(2)
                with col_pred:
                    label = "📈 UPWARD TREND" if prediction == 1 else "📉 DOWNWARD TREND"
                    color = "green" if prediction == 1 else "red"
                    st.markdown(f"### AI Prediction (Tomorrow): <span style='color:{color}'>{label}</span>", unsafe_allow_html=True)
                with col_conf:
                    st.metric("Model Confidence", f"{confidence:.1f}%")
            
            # --- 6. VISUALIZATION ---
            st.divider()
            st.subheader(f"Price History: {ticker}")
            st.line_chart(data['Close'].tail(days_to_show))
            
        else:
            st.warning("Not enough data to calculate indicators. Try a longer history.")
    else:
        st.error(f"❌ No data found for ticker: {ticker}")

except Exception as e:
    st.error(f"⚠️ System Error: {e}")
