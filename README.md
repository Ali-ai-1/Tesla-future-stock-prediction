#Stock Price Prediction using Machine Learning

This project demonstrates a short-term stock price prediction system using historical stock market data. The objective is to predict the **next day's closing price** of a selected stock (e.g., Tesla - TSLA) based on key market indicators like `Open`, `High`, `Low`, and `Volume`.

---

##Project Overview

-Load historical stock data using the [Yahoo Finance API](https://pypi.org/project/yfinance/)
-Train machine learning models using:
  - Random Forest Regressor
  - Linear Regression (for comparison)
-Predict and visualize the stock’s closing price
-Predict the next day's closing price using the latest available data

---

## 🛠️ Tech Stack & Libraries

- **Python 3**
- `yfinance` – to fetch stock data from Yahoo Finance
- `pandas` & `numpy` – for data handling
- `scikit-learn` – for ML models
- `matplotlib` – for data visualization

---

## 📁 Folder Structure


---

##Features Used for Training

| Feature  | Description                        |
|----------|------------------------------------|
| Open     | Opening price of the stock         |
| High     | Highest price during the day       |
| Low      | Lowest price during the day        |
| Volume   | Number of shares traded            |

**Target:**
- `Close` – The stock's closing price (what we predict)

---

##Sample Output

![Sample Chart](https://i.imgur.com/NWzBdGJ.png) <!-- Optional: Replace with your own output graph -->

---

##Installation

1. Clone the repo:
```bash
git clone https://github.com/yourusername/stock-price-prediction.git
cd stock-price-prediction

