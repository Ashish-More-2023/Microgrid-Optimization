"""
Step 5 - Data-driven forecasting.

Forecasts feed the Model Predictive Controller. We compare three models on a
chronological train/test split:
  * Persistence   : next value = current value
  * Climatology   : mean of all previous days at the same 15-min slot (causal)
  * Random Forest : gradient-free ML model with calendar + lag features
The comparison is also a diagnostic: if ML cannot beat climatology, the series
contains no learnable temporal structure.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def causal_climatology(series: pd.Series) -> pd.Series:
    """Mean of all strictly-earlier observations in the same 15-min slot of the day."""
    slot = series.index.hour * 4 + series.index.minute // 15
    clim = series.groupby(slot).transform(lambda s: s.shift(1).expanding().mean())
    fallback = series.shift(1).expanding().mean()
    return clim.fillna(fallback).fillna(series.iloc[0])


def _features(series: pd.Series) -> pd.DataFrame:
    X = pd.DataFrame(index=series.index)
    h = series.index.hour + series.index.minute / 60
    X["sin_h"] = np.sin(2 * np.pi * h / 24)
    X["cos_h"] = np.cos(2 * np.pi * h / 24)
    X["dow"] = series.index.dayofweek
    for lag in (1, 2, 4, 96):
        X[f"lag_{lag}"] = series.shift(lag)
    X["roll4"] = series.shift(1).rolling(4).mean()
    X["clim"] = causal_climatology(series)
    return X


def evaluate_forecasters(series: pd.Series, name: str, train_frac=0.75, seed=42):
    """One-step-ahead (15-min) forecast comparison. Returns (metrics_df, predictions_df)."""
    X = _features(series)
    data = pd.concat([X, series.rename("y")], axis=1).dropna()
    split = int(len(data) * train_frac)
    tr, te = data.iloc[:split], data.iloc[split:]

    rf = RandomForestRegressor(n_estimators=200, min_samples_leaf=20, n_jobs=-1, random_state=seed)
    rf.fit(tr.drop(columns="y"), tr["y"])

    preds = pd.DataFrame({
        "actual": te["y"],
        "persistence": te["lag_1"],
        "climatology": te["clim"],
        "random_forest": rf.predict(te.drop(columns="y")),
    })
    rows = []
    for model in ["persistence", "climatology", "random_forest"]:
        rows.append({
            "series": name, "model": model,
            "MAE_kW": mean_absolute_error(preds.actual, preds[model]),
            "RMSE_kW": mean_squared_error(preds.actual, preds[model]) ** 0.5,
            "R2": r2_score(preds.actual, preds[model]),
        })
    return pd.DataFrame(rows), preds
