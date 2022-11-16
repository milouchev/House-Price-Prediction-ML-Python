# Forecasting Boston, MA house prices using Machine Learning in Python

import pandas as pd

# US Economic Data (30Y fixed mortgage rate, rental vancancy rate, inflation rate)
stlouis_fed_files = ["MORTGAGE30US.csv", "RRVRUSQ156N.csv", "CPIAUCSL.csv"]
dfs = [pd.read_csv(f, parse_dates=True, index_col=0) for f in stlouis_fed_files]

# 30Y fixed mortgage interest rate data is weekly by default (MORTGAGE30US.csv)
print(dfs[0])
# Rental vacancy rate data is quarterly by default (RRVRUSQ156N.csv)
print(dfs[1])
# Inflation rate data is monthly by default (CPIAUCSL.csv)
print(dfs[2])

# Merging data
fed_data = pd.concat(dfs, axis=1)

# Forward filling data to remove NaN values... assuming rates are constant throughout the period they're in
fed_data = fed_data.ffill().dropna()

# Zillow US House Data (Median sale price of houses, Zillow-computed house value)
zillow_files = ["Metro_median_sale_price_uc_sfrcondo_week.csv", "Metro_zhvi_uc_sfrcondo_tier_0.33_0.67_month.csv"]

# Isolating the Boston, MA row in the data (row 11), and removing first 5 columns to have weekly sales price data only
dfs = [pd.read_csv(f) for f in zillow_files]
dfs = [pd.DataFrame(df.iloc[11,5:]) for df in dfs]
print(dfs[0])
# Now this format is consistent with the US Economic Data files

# Merging data using common column (month) that we can join both DataFrames on
from datetime import timedelta
for df in dfs:
    df.index = pd.to_datetime(df.index)
    df["month"] = df.index.to_period("M")
print(dfs[0])

price_data = dfs[0].merge(dfs[1], on="month")
price_data.index = dfs[0].index
print(price_data)

# Removing month column and assigning new names to variables
del price_data["month"]
price_data.columns = ["sales_price", "home_value"]

# Aligning fed_data dates with zillow_files dates to be able to merge everything
fed_data.index = fed_data.index + timedelta(days=2)

# Merging all DataFrames, using the index in both DataFrames to merge them
price_data = fed_data.merge(price_data, left_index=True, right_index=True)
price_data.columns = ["interest_rate", "vacancy_rate", "cpi", "sales_price", "home_value"]
print(price_data)

# Adjusting house prices for inflation
price_data["adj_sales_price"] = price_data["sales_price"] / price_data["cpi"] * 100

# Adjusting Zillow-computed house value for inflation
price_data["adj_home_value"] = price_data["home_value"] / price_data["cpi"] * 100

# Plotting sales_price and adj_sales_price over time
price_data.plot.line(y="sales_price", use_index=True)
price_data.plot.line(y="adj_sales_price", use_index=True)

# Setting up target for ML model, what house prices will look like next quarter
# Pandas shift will take adjusted price from 3 months into the future and bring to current row
price_data["next_quarter"] = price_data["adj_sales_price"].shift(-13)

# Dropping NaN values
price_data.dropna(inplace=True)
print(price_data)

# Counting house price changes... True = price change is positive, False = price change is negative
price_data["change"] = (price_data["next_quarter"] > price_data["adj_sales_price"]).astype(bool)
print(price_data["change"].value_counts())

# Setting up ML model (Random Forest)
predictors = ["interest_rate", "vacancy_rate", "adj_sales_price", "adj_home_value"]
target = "change"
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import numpy as np

# Weeks
START = 260
STEP = 52

def predict(train, test, predictors, target):
    rf = RandomForestClassifier(random_state=1)
    rf.fit(train[predictors], train[target])
    preds = rf.predict(test[predictors])
    return preds

def backtest(data, predictors, target):
    all_preds = []
    for i in range(START, data.shape[0], STEP):
        train = price_data.iloc[:i]
        test = price_data.iloc[i:(i+STEP)]
        all_preds.append(predict(train, test, predictors, target))
    preds = np.concatenate(all_preds)
    return preds, accuracy_score(data.iloc[START:][target], preds)

preds, accuracy = backtest(price_data, predictors, target)
accuracy = round(accuracy*100,2)
print("Accuracy: " + str(accuracy) + "%")

# Comparing predicted values to actual values
pred_match = (preds == price_data[target].iloc[START:])
pred_match[pred_match == True] = "green"
pred_match[pred_match == False] = "red"

# Plotting data - green data points are correct predictions, red data points are incorrect predictions
import matplotlib.pyplot as plt
plot_data = price_data.iloc[START:].copy()
plot_data.reset_index().plot.scatter(x="index", y="adj_sales_price", color=pred_match)
plt.title("Home Price Forecast - Boston, MA - Version 1")
plt.xlabel("Time")
plt.ylabel("Adjusted Home Sales Price ($USD)")

# Improving the accuracy of the ML model with additional variables
yearly = price_data.rolling(52, min_periods=1).mean()
yearly_ratios = [p + "_year" for p in predictors]
price_data[yearly_ratios] = price_data[predictors] / yearly[predictors]

preds, accuracy = backtest(price_data, predictors + yearly_ratios, target)
accuracy = round(accuracy*100,2)
print("Accuracy: " + str(accuracy) + "%")

# Plotting data again
pred_match = (preds == price_data[target].iloc[START:])
pred_match[pred_match == True] = "green"
pred_match[pred_match == False] = "red"
plot_data = price_data.iloc[START:].copy()
plot_data.reset_index().plot.scatter(x="index", y="adj_sales_price", color=pred_match)
plt.title("Home Price Forecast - Boston, MA - Version 2")
plt.xlabel("Time")
plt.ylabel("Adjusted Home Sales Price ($USD)")