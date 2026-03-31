📈 AI-Powered Market Sentiment & Trend Predictor
A full-stack Machine Learning application that fetches real-time financial data, processes technical indicators, and uses a Random Forest Classifier to predict stock direction with built-in confidence metrics.

🔗 Live Application
https://ai-market-sentiment-predictor-mbbjqunmnmqidrccqnnn5l.streamlit.app/

🌟 Why This Project Matters
Financial markets are noisy. This tool helps cut through the noise by combining:

Quantitative Data: 50-day and 10-day Moving Averages + Daily Returns.

Qualitative Data: Real-time news sentiment analysis via NLP.

Probabilistic AI: Instead of a simple "Up/Down," the model provides a Confidence Percentage to help gauge risk.

🛠️ The Tech Stack
Backend: Python 3.12

Machine Learning: Scikit-Learn (Random Forest), Joblib

Data API: Yahoo Finance (yfinance)

NLP: TextBlob (Sentiment Polarity)

Frontend: Streamlit Framework

Deployment: GitHub & Streamlit Community Cloud

📊 How I Solved Key Engineering Challenges
Data Flattening: Handled yfinance Multi-Index header issues to ensure the model works seamlessly across different tickers (AAPL, TSLA, NVDA).

Feature Engineering: Implemented automated calculation of rolling averages and percentage changes on live-streamed data.

Error Resilience: Built a "Nuclear Option" data cleaning pipeline to handle missing values and API inconsistencies.

⚙️ How to Run Locally
Clone the Repo:

Bash
git clone https://github.com/devashreepalav/AI-Market-Sentiment-Predictor.git
cd AI-Market-Sentiment-Predictor
Install Dependencies:

Bash
pip install -r requirements.txt
Launch the Dashboard:

Bash
streamlit run app.py
📁 Project Structure
app.py: The main UI and logic for the live dashboard.

main.ipynb: The research and training phase of the model.

stock_predictor_model.pkl: The saved AI weights (The "Brain").

requirements.txt: List of all Python libraries needed.
