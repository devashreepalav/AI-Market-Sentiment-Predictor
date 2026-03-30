import streamlit as st
import yfinance as yf
import joblib
import pandas as pd
from newsapi import NewsApiClient
from textblob import TextBlob

# --- 1. SETTINGS & API CONFIG ---
# Replace with your actual NewsAPI key: https://newsapi.org/
NEWS_API_KEY = 'YOUR_NEWS_API_KEY_HERE' 
newsapi = NewsApiClient(api_key=NEWS_API_KEY)

st.set_page_config(page_title="AI Market Vibe", layout="wide", page_icon="📈")

# --- 2. LOAD TRAINED MODEL ---
@st.cache_resource
def load_ai_model():
    try:
        model = joblib.load('stock_predictor_model.pkl')
        features = joblib.load('model_features.pkl')
        return model, features
    except FileNotFoundError:
        return None, None

model, model_features = load_ai_model()

# --- 3. SIDEBAR & LOGIC ---
st.sidebar.header("🕹️ Control Panel")
ticker = st.sidebar.text_input("Enter Stock Ticker", "AAPL").upper()
days_to_show = st.sidebar.slider("Days of History", 30, 365, 60)

st.title(f"🚀 AI Market Insight: {ticker}")

# --- 4. FETCH DATA ---
try:
    # Use auto_adjust=True to help standardize columns
    data = yf.download(ticker, period=f"{days_to_show+50}d", interval="1d", auto_adjust=True)
    
    if not data.empty:
        # 🔥 THE CRITICAL FIX: Strip ticker names from multi-index columns
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = [col[0] if isinstance(col, tuple) else col for col in data.columns]
        else:
            # Even if it's not multi-index, force standard names
            data.columns = data.columns.str.capitalize()

        # Ensure we have the basic columns needed
        required_cols = ['Close', 'Volume']
        if all(col in data.columns for col in required_cols):
            # Feature Engineering
            data['Returns'] = data['Close'].pct_change()
            data['MA10'] = data['Close'].rolling(window=10).mean()
            data['MA50'] = data['Close'].rolling(window=50).mean()
            
            # Create latest_row
            latest_row = data.tail(1)
            
            # --- 5. PREDICTION UI ---
            if model is not None and not latest_row.isnull().values.any():
                # Re-order and select only the features the model was trained on
                inputs = latest_row[model_features]
                prediction = model.predict(inputs)[0]
                confidence = model.predict_proba(inputs).max() * 100
                
                col_pred, col_conf = st.columns(2)
                with col_pred:
                    label = "📈 UPWARD TREND" if prediction == 1 else "📉 DOWNWARD TREND"
                    color = "green" if prediction == 1 else "red"
                    st.markdown(f"### AI Prediction (Tomorrow): <span style='color:{color}'>{label}</span>", unsafe_allow_html=True)
                with col_conf:
                    st.metric("Model Confidence", f"{confidence:.1f}%")
        else:
            st.error(f"Missing required columns. Found: {list(data.columns)}")
    else:
        st.error(f"❌ No data found for ticker: {ticker}")

except Exception as e:
    st.error(f"⚠️ Error: {e}")
    # --- 6. VISUALIZATION ---
    st.divider()
    col_chart, col_news = st.columns([2, 1])
    
    with col_chart:
        st.subheader("Price History & Trends")
        st.line_chart(data['Close'].tail(days_to_show))
        
    with col_news:
        st.subheader("Latest News Sentiment")
        try:
            articles = newsapi.get_everything(q=ticker, language='en', sort_by='publishedAt')
            if articles['articles']:
                for art in articles['articles'][:5]:
                    score = TextBlob(art['title']).sentiment.polarity
                    mood = "🟢" if score > 0 else "🔴" if score < 0 else "⚪"
                    st.write(f"{mood} [{art['title']}]({art['url']})")
            else:
                st.write("No recent news found.")
        except:
            st.write("Check your NewsAPI Key in the code!")

else:
    st.error("Invalid Ticker or No Data Found.")
