Key Features
Predictive Modeling: Uses a Random Forest Classifier to analyze technical indicators and predict the next day's price direction (Up/Down).

NLP Sentiment Analysis: Integrates NewsAPI and TextBlob to quantify the "mood" of real-time financial headlines.

Interactive Dashboard: Built with Streamlit to allow users to toggle between different tickers (e.g., AAPL, TSLA, NVDA).

Confidence Scoring: Provides a percentage-based confidence level for every AI prediction using predict_proba.

🛠️ Tech Stack
Language: Python 3.12

Libraries: Scikit-Learn, Pandas, NumPy, Joblib, YFinance

UI/UX: Streamlit

APIs: NewsAPI (Sentiment Data), Yahoo Finance (Market Data)

📊 How It Works
Data Ingestion: The app pulls 2 years of historical price data and the latest 10 news headlines for a given ticker.

Feature Engineering: Calculates 10-day and 50-day Moving Averages, Daily Returns, and Volatility.

Sentiment Scoring: Financial headlines are processed into a polarity score (-1 to 1) to provide context for price movements.

Inference: The pre-trained Random Forest model (stored in .pkl format) processes the technical data to output a final prediction.