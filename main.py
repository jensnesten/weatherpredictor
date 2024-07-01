#%%
import pandas as pd
import math 
from sklearn.linear_model import Ridge
weather = pd.read_csv("./Data/weather.csv", index_col="DATE")

null_pct = weather.apply(pd.isnull).sum()/weather.shape[0]
valid_col = weather.columns[null_pct < .05]
valid_col
weather = weather[valid_col].copy()
weather.columns = weather.columns.str.lower()
weather = weather.ffill()
weather.apply(pd.isnull).sum()

weather.index = pd.to_datetime(weather.index)
weather.index.year.value_counts().sort_index()

weather["target"] = weather.shift(-1)["tmax"]
weather = weather.ffill()
weather

rr = Ridge(alpha=.1)

def backtest(weather, model, predictors, start=1000, step=30):
    all_preds = []

    for i in range(start, weather.shape[0], step):
        train = weather.iloc[:i,:]
        test = weather.iloc[i:(i+step),:]
        model.fit(train[predictors], train["target"])
        preds = model.predict(test[predictors])

        preds = pd.Series(preds, index=test.index)
        combined = pd.concat([test["target"], preds], axis=1)

        combined.columns = ["actual", "prediction"]

        combined["diff"] = (combined["prediction"] - combined["actual"]).abs()

        all_preds.append(combined)
    return pd.concat(all_preds)


from sklearn.metrics import mean_absolute_error

def pct_diff(old, new):
    return (new-old)/old


def compute_rolling(weather, horizon, col):
    label = f"rolling_{horizon}_{col}"

    weather[label] = weather[col].rolling(horizon).mean()
    weather[f"{label}_pct"] = pct_diff(weather[label], weather[col])
    return weather

rolling_horizons = [3, 7, 13, 28, 144]

for horizon in rolling_horizons:
    for col in ["tmax", "tmin", "prcp"]:
        weather = compute_rolling(weather, horizon, col)

weather = weather.iloc[14:,:]
weather = weather.fillna(0)

def expand_mean(df):
    return df.expanding(1).mean()

for col in ["tmax", "tmin", "prcp"]:
    weather[f"month_avg_{col}"] = weather[col].groupby(weather.index.month, group_keys=False).apply(expand_mean)
    weather[f"day_avg_{col}"] = weather[col].groupby(weather.index.day_of_year, group_keys=False).apply(expand_mean)
    weather[f"week_avg_{col}"] = weather[col].groupby(weather.index.week, group_keys=False).apply(expand_mean)

predictors = weather.columns[~weather.columns.isin(["target", "name", "station"])]
predictors
# %%

predictions = backtest(weather, rr, predictors)
print(predictions)
std_d = (mean_absolute_error(predictions["actual"], predictions["prediction"]) * 5) / 9 
print(std_d)

# %%
print(weather.corr())

# %%
