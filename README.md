# 📈 AI Market Insight & Sentiment Engine

An end-to-end Machine Learning application that combines **Natural Language Processing (NLP)** and **Supervised Learning** to predict stock trends and visualize market sentiment.

## 🔗 Live Demo
http://localhost:8501/

## 🚀 Key Features
* **Predictive Modeling:** Uses a **Random Forest Classifier** to analyze technical indicators and predict the next day's price direction (Up/Down).
* **NLP Sentiment Analysis:** Integrates **NewsAPI** and **TextBlob** to quantify the "mood" of real-time financial headlines.
* **Interactive Dashboard:** Built with **Streamlit** to allow users to toggle between different tickers (e.g., AAPL, TSLA, NVDA).
* **Confidence Scoring:** Provides a percentage-based confidence level for every AI prediction using `predict_proba`.

## 🛠️ Tech Stack
* **Language:** Python 3.12
* **Libraries:** Scikit-Learn, Pandas, NumPy, Joblib, YFinance
* **UI/UX:** Streamlit
* **APIs:** NewsAPI (Sentiment Data), Yahoo Finance (Market Data)

## 📊 How It Works
1. **Data Ingestion:** The app pulls 2 years of historical price data and the latest 10 news headlines for a given ticker.
2. **Feature Engineering:** Calculates 10-day and 50-day **Moving Averages (MA)**, **Daily Returns**, and **Volatility**.
3. **Sentiment Scoring:** Financial headlines are processed into a polarity score (-1 to 1) to provide context for price movements.
4. **Inference:** The pre-trained Random Forest model (stored in `.pkl` format) processes the technical data to output a final prediction.

## ⚙️ Installation & Setup
1. Clone this repository:
   ```bash
   git clone [https://github.com/YOUR_USERNAME/AI-Stock-Predictor.git](https://github.com/devashreepalav/AI-Stock-Predictor.git)
