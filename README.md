# 📈 AI-Powered Market Sentiment & Trend Predictor

An end-to-end Machine Learning application that fetches real-time financial data, processes technical indicators, and uses a **Random Forest Classifier** to predict stock direction with built-in confidence metrics.

---

## 🔗 Live Application
### **[🚀 Click Here to View the Live App]
(https://ai-market-sentiment-predictor-mbbjqunmnmqidrccqnnn5l.streamlit.app/)**

---

## 🌟 Why This Project Matters
Financial markets are noisy. This tool helps cut through the noise by combining:
* **Quantitative Data:** 50-day and 10-day Moving Averages + Daily Returns.
* **Qualitative Data:** Real-time news sentiment analysis via NLP.
* **Probabilistic AI:** Instead of a simple "Up/Down," the model provides a **Confidence Percentage** to help gauge risk.

## 🛠️ Tech Stack
* **Backend:** Python 3.12
* **Machine Learning:** Scikit-Learn (Random Forest)
* **Data API:** Yahoo Finance (`yfinance`)
* **NLP:** TextBlob (Sentiment Polarity)
* **Frontend:** Streamlit Framework
* **Deployment:** GitHub & Streamlit Community Cloud

## 📊 How It Works
1. **Data Ingestion:** Fetches 2 years of price data and latest headlines via API.
2. **Feature Engineering:** Handles `yfinance` Multi-Index structures to calculate rolling averages.
3. **Inference:** Uses a pre-trained `.pkl` model to generate predictions on live data.

## ⚙️ Setup & Installation
1. **Clone the repo:** `git clone https://github.com/devashreepalav/AI-Market-Sentiment-Predictor.git`
2. **Install requirements:** `pip install -r requirements.txt`
3. **Run app:** `streamlit run app.py`
