# Stock Price Direction Prediction Model

A machine learning project that predicts the next day's directional movement (Up/Down) of a stock using technical analysis indicators and a Random Forest classifier.
*THIS IS JUST A SAMPLE MODEL WITH NO CURRENT WORKING APPLICATION USEFUL ONLY AS A REFERENCE FOR KNOWN THINGS AND CONTAINS ONLY CODE ; FILES MUST BE CREATED ON OWN FOR THE USE OF CODE *

## 🎯 Objective

The goal of this project is not to build a profitable trading system but to demonstrate a end-to-end data science workflow:
*   Data acquisition from a financial API
*   Feature engineering to create predictive technical indicators
*   Machine learning model training and evaluation
*   Results visualization and interpretation

## ⚙️ Installation & Setup

1.  **Clone the repository:**
    ```bash
    git clone (https://github.com/RIshi-Raj-noobie/Stock-market-predictor)
    cd stock_prediction_project
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## 🚀 Usage

1.  Launch Jupyter Notebook:
    ```bash
    jupyter notebook
    ```
2.  Open and run the `stock_analysis.ipynb` notebook cell-by-cell.
3.  The notebook will download data for Apple Inc. (AAPL) by default. You can change the `ticker` variable to analyze a different stock (e.g., `ticker = 'MSFT'`).

## 📊 Methodology

### 1. Data Collection
Historical OHLCV (Open, High, Low, Close, Volume) data is fetched using the `yfinance` library which pulls data from Yahoo Finance.

### 2. Feature Engineering
The following technical indicators were created from the raw price data:
*   **Simple Moving Averages (SMA_20, SMA_50):** Identify trends.
*   **Price Rate of Change (ROC):** Measures momentum.
*   **Volatility:** Standard deviation of closing prices, measuring risk.
*   **Relative Strength Index (RSI):** Identifies overbought/oversold conditions.

### 3. Target Variable
The target variable (`y`) is a binary label:
*   **1** if the next day's closing price is higher than the current day's close.
*   **0** if it is lower or equal.

### 4. Model
A **Random Forest Classifier** was chosen for its ability to handle non-linear relationships and provide feature importance metrics. Features were standardized before training.

## 📈 Results

The model's performance is evaluated on a held-out test set. Key metrics include:
*   **Accuracy:** The percentage of correct directional predictions.
*   **Classification Report:** Precision, Recall, and F1-Score for both classes.
*   **Confusion Matrix:** A breakdown of true vs. predicted values.
*   **Feature Importance:** Shows which indicators were most influential in the model's decisions.

*Note: A high accuracy (e.g., >55%) in financial prediction is notoriously difficult due to market efficiency and noise. The primary value of this project is the demonstration of technical skills.*

## 🛠️ Technologies Used

*   **Programming Language:** Python
*   **Libraries:** Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn, yfinance
*   **Environment:** Jupyter Notebook


