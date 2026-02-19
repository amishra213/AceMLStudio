"""
AceML Studio – Time Series Analysis & Forecasting
====================================================
Comprehensive time series module supporting:
  • Auto-detection and parsing of datetime columns
  • Stationarity testing (ADF, KPSS)
  • Seasonal decomposition (additive / multiplicative)
  • Classical models: ARIMA / SARIMA, Exponential Smoothing (Holt-Winters)
  • ML-based: Prophet (optional), feature-based regression
  • Forecasting with confidence intervals
  • Trend / seasonality / residual analysis
  • Multi-step ahead forecasting
"""

import logging
import time
import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger("aceml.time_series")

# ── Optional heavy imports (graceful degradation) ─────────────────
try:
    from statsmodels.tsa.stattools import adfuller, kpss
    from statsmodels.tsa.seasonal import seasonal_decompose
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    from statsmodels.tsa.arima.model import ARIMA as StatsARIMA
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    from statsmodels.graphics.tsaplots import acf, pacf
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False
    logger.info("statsmodels not installed – classical TS models disabled")

try:
    from prophet import Prophet  # type: ignore
    HAS_PROPHET = True
except ImportError:
    HAS_PROPHET = False
    logger.info("Prophet not installed – Prophet forecasting disabled")

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=FutureWarning)


# ════════════════════════════════════════════════════════════════════
#  Time-Series Engine
# ════════════════════════════════════════════════════════════════════

class TimeSeriesEngine:
    """Full-lifecycle time-series analysis and forecasting."""

    # ----------------------------------------------------------------
    #  Datetime Detection & Preparation
    # ----------------------------------------------------------------
    @staticmethod
    def detect_datetime_columns(df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Auto-detect columns that look like datetime values."""
        candidates: List[Dict[str, Any]] = []
        for col in df.columns:
            # Already datetime
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                candidates.append({
                    "column": col,
                    "dtype": str(df[col].dtype),
                    "detected_as": "datetime64",
                    "sample": str(df[col].dropna().iloc[0]) if len(df[col].dropna()) else None,
                    "null_count": int(df[col].isna().sum()),
                })
                continue
            # Object / string columns – try parsing
            if df[col].dtype == object:
                sample = df[col].dropna()
                if len(sample) == 0:
                    continue
                try:
                    parsed = pd.to_datetime(sample.head(50), errors="coerce")
                    success_rate = parsed.notna().mean()
                    if success_rate >= 0.8:
                        candidates.append({
                            "column": col,
                            "dtype": "object (parseable)",
                            "detected_as": "string_datetime",
                            "success_rate": round(float(success_rate), 2),
                            "sample": str(sample.iloc[0]),
                            "null_count": int(df[col].isna().sum()),
                        })
                except Exception:
                    pass
            # Numeric columns that could be Unix timestamps
            if pd.api.types.is_numeric_dtype(df[col]):
                sample = df[col].dropna()
                if len(sample) == 0:
                    continue
                mn, mx = float(sample.min()), float(sample.max())
                # Reasonable Unix timestamp range: 2000-01-01 to 2040-01-01
                if 9.46e8 < mn and mx < 2.21e9:
                    candidates.append({
                        "column": col,
                        "dtype": "numeric (unix timestamp)",
                        "detected_as": "unix_timestamp",
                        "min_val": mn,
                        "max_val": mx,
                        "sample": str(sample.iloc[0]),
                        "null_count": int(df[col].isna().sum()),
                    })
        logger.info("Detected %d datetime candidate columns", len(candidates))
        return candidates

    @staticmethod
    def prepare_time_series(
        df: pd.DataFrame,
        datetime_col: str,
        value_col: str,
        freq: Optional[str] = None,
        fill_method: str = "ffill",
    ) -> pd.DataFrame:
        """
        Prepare a DataFrame for time-series analysis.
        Returns a DataFrame indexed by datetime with the value column.
        """
        ts_df = df[[datetime_col, value_col]].copy()

        # Parse datetime
        if not pd.api.types.is_datetime64_any_dtype(ts_df[datetime_col]):
            ts_df[datetime_col] = pd.to_datetime(ts_df[datetime_col], errors="coerce")
        ts_df = ts_df.dropna(subset=[datetime_col])
        ts_df = ts_df.sort_values(datetime_col)
        ts_df = ts_df.set_index(datetime_col)

        # Infer frequency if not provided
        if freq is None:
            inferred = pd.infer_freq(pd.DatetimeIndex(ts_df.index))
            if inferred:
                freq = inferred
                logger.info("Inferred frequency: %s", freq)
            else:
                # Fallback: estimate from median time delta
                deltas = ts_df.index.to_series().diff().dropna()
                if len(deltas) > 0:
                    median_delta = pd.Timedelta(deltas.median())  # type: ignore[arg-type]
                    if median_delta <= pd.Timedelta(hours=1):
                        freq = "h"
                    elif median_delta <= pd.Timedelta(days=1):
                        freq = "D"
                    elif median_delta <= pd.Timedelta(days=7):
                        freq = "W"
                    else:
                        freq = "MS"
                    logger.info("Estimated frequency from median delta: %s", freq)
                else:
                    freq = "D"

        ts_df = ts_df.asfreq(freq)

        # Fill gaps
        if fill_method == "ffill":
            ts_df = ts_df.ffill()
        elif fill_method == "bfill":
            ts_df = ts_df.bfill()
        elif fill_method == "interpolate":
            ts_df = ts_df.interpolate(method="time")
        elif fill_method == "zero":
            ts_df = ts_df.fillna(0)

        # Drop remaining NaN
        ts_df = ts_df.dropna()
        logger.info("Prepared time series: %d rows, freq=%s", len(ts_df), freq)
        return ts_df

    # ----------------------------------------------------------------
    #  Stationarity Tests
    # ----------------------------------------------------------------
    @staticmethod
    def stationarity_test(series: pd.Series) -> Dict[str, Any]:
        """Run ADF and KPSS stationarity tests."""
        if not HAS_STATSMODELS:
            return {"error": "statsmodels not installed"}

        result: Dict[str, Any] = {}
        values = series.dropna().values.astype(float)

        # Augmented Dickey-Fuller
        try:
            adf_result: Any = adfuller(values, autolag="AIC")
            adf_stat = float(adf_result[0])
            adf_p    = float(adf_result[1])
            adf_lags = int(adf_result[2])
            adf_nobs = int(adf_result[3])
            adf_crit = dict(adf_result[4])
            result["adf"] = {
                "statistic": round(adf_stat, 4),
                "p_value": round(adf_p, 6),
                "lags_used": adf_lags,
                "n_observations": adf_nobs,
                "critical_values": {k: round(float(v), 4) for k, v in adf_crit.items()},
                "is_stationary": adf_p < 0.05,
            }
        except Exception as e:
            result["adf"] = {"error": str(e)}

        # KPSS
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                kpss_result = kpss(values, regression="c", nlags="auto")
            kpss_stat: float = float(kpss_result[0])
            kpss_p:    float = float(kpss_result[1])
            kpss_lags: int   = int(kpss_result[2])
            kpss_crit: dict  = dict(kpss_result[3])
            result["kpss"] = {
                "statistic": round(kpss_stat, 4),
                "p_value": round(kpss_p, 6),
                "lags_used": kpss_lags,
                "critical_values": {k: round(float(v), 4) for k, v in kpss_crit.items()},
                "is_stationary": kpss_p > 0.05,
            }
        except Exception as e:
            result["kpss"] = {"error": str(e)}

        # Summary
        adf_stat_flag = result.get("adf", {}).get("is_stationary", None)
        kpss_stat_flag = result.get("kpss", {}).get("is_stationary", None)
        if adf_stat_flag is True and kpss_stat_flag is True:
            result["conclusion"] = "stationary"
        elif adf_stat_flag is False and kpss_stat_flag is False:
            result["conclusion"] = "non_stationary"
        elif adf_stat_flag is True and kpss_stat_flag is False:
            result["conclusion"] = "trend_stationary"
        elif adf_stat_flag is False and kpss_stat_flag is True:
            result["conclusion"] = "difference_stationary"
        else:
            result["conclusion"] = "inconclusive"

        result["recommendation"] = {
            "stationary": "Series is stationary — can proceed with modeling.",
            "non_stationary": "Series is non-stationary — apply differencing (d≥1) before ARIMA.",
            "trend_stationary": "Series is trend-stationary — consider detrending or include a trend term.",
            "difference_stationary": "Series is difference-stationary — first differencing recommended.",
            "inconclusive": "Results are mixed — try differencing and re-test.",
        }.get(result["conclusion"], "")

        return result

    # ----------------------------------------------------------------
    #  Autocorrelation
    # ----------------------------------------------------------------
    @staticmethod
    def autocorrelation(series: pd.Series, nlags: int = 40) -> Dict[str, Any]:
        """Compute ACF and PACF values."""
        if not HAS_STATSMODELS:
            return {"error": "statsmodels not installed"}
        values = series.dropna().values.astype(float)
        nlags = min(nlags, len(values) // 2 - 1)
        if nlags < 1:
            return {"error": "Not enough data points for autocorrelation"}
        acf_vals = acf(values, nlags=nlags, fft=True)
        pacf_vals = pacf(np.asarray(values, dtype=float), nlags=nlags, method="ywm")
        return {
            "acf": [round(float(v), 4) for v in acf_vals],
            "pacf": [round(float(v), 4) for v in pacf_vals],
            "nlags": nlags,
            "confidence_interval": round(1.96 / np.sqrt(len(values)), 4),
        }

    # ----------------------------------------------------------------
    #  Seasonal Decomposition
    # ----------------------------------------------------------------
    @staticmethod
    def decompose(
        series: pd.Series,
        model: str = "additive",
        period: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Decompose series into trend, seasonal, and residual."""
        if not HAS_STATSMODELS:
            return {"error": "statsmodels not installed"}
        values = series.dropna()
        if period is None:
            period = min(len(values) // 2, 12)  # default guess
        if period < 2:
            period = 2
        try:
            decomposition = seasonal_decompose(values, model=model, period=period)
            # Safely convert, replacing NaN with None
            def _safe_list(arr: Any) -> list:
                return [None if (v is None or (isinstance(v, float) and np.isnan(v))) else round(float(v), 4) for v in arr]

            return {
                "model": model,
                "period": period,
                "trend": _safe_list(decomposition.trend),
                "seasonal": _safe_list(decomposition.seasonal),
                "residual": _safe_list(decomposition.resid),
                "observed": _safe_list(decomposition.observed),
                "dates": [str(d) for d in values.index],
            }
        except Exception as e:
            logger.error("Decomposition failed: %s", e)
            return {"error": str(e)}

    # ----------------------------------------------------------------
    #  Summary Statistics for Time Series
    # ----------------------------------------------------------------
    @staticmethod
    def ts_summary(series: pd.Series) -> Dict[str, Any]:
        """Comprehensive time-series summary statistics."""
        values = series.dropna()
        result: Dict[str, Any] = {
            "count": int(len(values)),
            "mean": round(float(values.mean()), 4),
            "std": round(float(values.std()), 4),
            "min": round(float(values.min()), 4),
            "max": round(float(values.max()), 4),
            "median": round(float(values.median()), 4),
            "skewness": round(float(values.skew()), 4),  # type: ignore[arg-type]
            "kurtosis": round(float(values.kurtosis()), 4),  # type: ignore[arg-type]
        }
        # Rolling stats
        if len(values) >= 10:
            window = max(len(values) // 10, 2)
            rolling_mean = values.rolling(window=window).mean().dropna()
            rolling_std = values.rolling(window=window).std().dropna()
            result["rolling_mean_trend"] = "increasing" if rolling_mean.iloc[-1] > rolling_mean.iloc[0] else "decreasing"
            result["rolling_std_trend"] = "increasing" if rolling_std.iloc[-1] > rolling_std.iloc[0] else "decreasing"
            result["rolling_window"] = window
        # Date range
        if hasattr(values.index, "min") and hasattr(values.index, "max"):
            try:
                result["date_range"] = {
                    "start": str(values.index.min()),
                    "end": str(values.index.max()),
                }
            except Exception:
                pass
        return result

    # ================================================================
    #  FORECASTING MODELS
    # ================================================================

    # ----------------------------------------------------------------
    #  ARIMA / SARIMA
    # ----------------------------------------------------------------
    @staticmethod
    def fit_arima(
        series: pd.Series,
        order: Tuple[int, int, int] = (1, 1, 1),
        seasonal_order: Optional[Tuple[int, int, int, int]] = None,
        forecast_steps: int = 30,
    ) -> Dict[str, Any]:
        """Fit ARIMA or SARIMA model and produce forecast."""
        if not HAS_STATSMODELS:
            return {"error": "statsmodels not installed"}

        values = series.dropna().astype(float)
        start_time = time.time()

        try:
            if seasonal_order:
                model = SARIMAX(values, order=order, seasonal_order=seasonal_order,
                                enforce_stationarity=False, enforce_invertibility=False)
                model_type = "SARIMA"
            else:
                model = StatsARIMA(values, order=order)
                model_type = "ARIMA"

            import warnings as _w
            with _w.catch_warnings():
                _w.simplefilter("ignore")
                fitted: Any = model.fit()
            forecast_result = fitted.get_forecast(steps=forecast_steps)
            forecast_mean = forecast_result.predicted_mean
            conf_int = forecast_result.conf_int(alpha=0.05)

            # In-sample predictions for evaluation
            in_sample = fitted.fittedvalues
            actual = values[in_sample.index]

            duration = round(time.time() - start_time, 2)
            logger.info("%s(%s) fitted in %.2fs", model_type, order, duration)

            return {
                "model_type": model_type,
                "order": list(order),
                "seasonal_order": list(seasonal_order) if seasonal_order else None,
                "training_time_sec": duration,
                "aic": round(float(fitted.aic), 2),
                "bic": round(float(fitted.bic), 2),
                "forecast": {
                    "values": [round(float(v), 4) for v in forecast_mean],
                    "lower_ci": [round(float(v), 4) for v in conf_int.iloc[:, 0]],
                    "upper_ci": [round(float(v), 4) for v in conf_int.iloc[:, 1]],
                    "dates": [str(d) for d in forecast_mean.index],
                    "steps": forecast_steps,
                },
                "metrics": {
                    "mae": round(float(mean_absolute_error(actual, in_sample)), 4),
                    "rmse": round(float(np.sqrt(mean_squared_error(actual, in_sample))), 4),
                    "r2": round(float(r2_score(actual, in_sample)), 4),
                },
                "residual_diagnostics": {
                    "mean": round(float(fitted.resid.mean()), 4),
                    "std": round(float(fitted.resid.std()), 4),
                },
                "summary": str(fitted.summary()),
            }
        except Exception as e:
            logger.error("ARIMA fitting failed: %s", e, exc_info=True)
            return {"error": str(e), "model_type": "ARIMA"}

    # ----------------------------------------------------------------
    #  Exponential Smoothing (Holt-Winters)
    # ----------------------------------------------------------------
    @staticmethod
    def fit_exponential_smoothing(
        series: pd.Series,
        trend: Optional[str] = "add",
        seasonal: Optional[str] = "add",
        seasonal_periods: Optional[int] = None,
        forecast_steps: int = 30,
    ) -> Dict[str, Any]:
        """Fit Holt-Winters Exponential Smoothing."""
        if not HAS_STATSMODELS:
            return {"error": "statsmodels not installed"}

        values = series.dropna().astype(float)
        start_time = time.time()

        # Guess seasonal periods
        if seasonal_periods is None:
            freq = getattr(values.index, "freqstr", None)
            if freq:
                freq_upper = freq.upper()
                if "D" in freq_upper:
                    seasonal_periods = 7
                elif "W" in freq_upper:
                    seasonal_periods = 52
                elif "M" in freq_upper or "MS" in freq_upper:
                    seasonal_periods = 12
                elif "Q" in freq_upper:
                    seasonal_periods = 4
                elif "H" in freq_upper:
                    seasonal_periods = 24
            if seasonal_periods is None:
                seasonal_periods = min(12, len(values) // 3)

        # Ensure enough data for seasonal decomposition
        if len(values) < 2 * seasonal_periods:
            seasonal = None
            logger.info("Not enough data for seasonal component – fitting without seasonality")

        try:
            model = ExponentialSmoothing(
                values,
                trend=trend,
                seasonal=seasonal,
                seasonal_periods=seasonal_periods if seasonal else None,
            )
            fitted = model.fit(optimized=True)
            forecast_vals = fitted.forecast(forecast_steps)

            # In-sample
            in_sample = fitted.fittedvalues
            actual = values[in_sample.index]

            # Confidence intervals (approximate via residual std)
            resid_std = float(fitted.resid.std())
            z = 1.96
            lower_ci = forecast_vals - z * resid_std
            upper_ci = forecast_vals + z * resid_std

            duration = round(time.time() - start_time, 2)
            model_label = "Holt-Winters"
            if trend and seasonal:
                model_label = f"Holt-Winters ({trend} trend, {seasonal} seasonal)"

            return {
                "model_type": "exponential_smoothing",
                "label": model_label,
                "trend": trend,
                "seasonal": seasonal,
                "seasonal_periods": seasonal_periods,
                "training_time_sec": duration,
                "aic": round(float(fitted.aic), 2),
                "bic": round(float(fitted.bic), 2),
                "forecast": {
                    "values": [round(float(v), 4) for v in forecast_vals],
                    "lower_ci": [round(float(v), 4) for v in lower_ci],
                    "upper_ci": [round(float(v), 4) for v in upper_ci],
                    "dates": [str(d) for d in forecast_vals.index],
                    "steps": forecast_steps,
                },
                "metrics": {
                    "mae": round(float(mean_absolute_error(actual, in_sample)), 4),
                    "rmse": round(float(np.sqrt(mean_squared_error(actual, in_sample))), 4),
                    "r2": round(float(r2_score(actual, in_sample)), 4),
                },
                "smoothing_params": {
                    "alpha": round(float(fitted.params.get("smoothing_level", 0)), 4),
                    "beta": round(float(fitted.params.get("smoothing_trend", 0)), 4),
                    "gamma": round(float(fitted.params.get("smoothing_seasonal", 0)), 4),
                },
            }
        except Exception as e:
            logger.error("Exponential Smoothing failed: %s", e, exc_info=True)
            return {"error": str(e), "model_type": "exponential_smoothing"}

    # ----------------------------------------------------------------
    #  Prophet
    # ----------------------------------------------------------------
    @staticmethod
    def fit_prophet(
        series: pd.Series,
        forecast_steps: int = 30,
        changepoint_prior_scale: float = 0.05,
        seasonality_mode: str = "additive",
        yearly_seasonality: Any = "auto",
        weekly_seasonality: Any = "auto",
        daily_seasonality: Any = "auto",
    ) -> Dict[str, Any]:
        """Fit Facebook Prophet model."""
        if not HAS_PROPHET:
            return {"error": "Prophet not installed. Install via: pip install prophet"}

        values = series.dropna().astype(float)
        start_time = time.time()

        try:
            # Prophet expects 'ds' and 'y' columns
            prophet_df = pd.DataFrame({
                "ds": values.index,
                "y": values.values,
            })

            model = Prophet(
                changepoint_prior_scale=changepoint_prior_scale,
                seasonality_mode=seasonality_mode,
                yearly_seasonality=yearly_seasonality,
                weekly_seasonality=weekly_seasonality,
                daily_seasonality=daily_seasonality,
            )
            model.fit(prophet_df)

            # Forecast
            freq = getattr(values.index, "freqstr", "D") or "D"
            future = model.make_future_dataframe(periods=forecast_steps, freq=freq)
            prediction = model.predict(future)

            forecast_portion = prediction.tail(forecast_steps)
            in_sample_pred = prediction.head(len(values))

            # Metrics
            metrics_dict: Dict[str, Any] = {}
            try:
                y_actual: np.ndarray = np.array(values.values, dtype=float)
                y_pred_in: np.ndarray = np.array(in_sample_pred["yhat"].values, dtype=float)
                metrics_dict = {
                    "mae": round(float(mean_absolute_error(y_actual, y_pred_in)), 4),
                    "rmse": round(float(np.sqrt(mean_squared_error(y_actual, y_pred_in))), 4),
                    "r2": round(float(r2_score(y_actual, y_pred_in)), 4),
                }
            except Exception:
                pass

            duration = round(time.time() - start_time, 2)

            return {
                "model_type": "prophet",
                "training_time_sec": duration,
                "changepoint_prior_scale": changepoint_prior_scale,
                "seasonality_mode": seasonality_mode,
                "forecast": {
                    "values": [round(float(v), 4) for v in forecast_portion["yhat"]],
                    "lower_ci": [round(float(v), 4) for v in forecast_portion["yhat_lower"]],
                    "upper_ci": [round(float(v), 4) for v in forecast_portion["yhat_upper"]],
                    "dates": [str(d) for d in forecast_portion["ds"]],
                    "steps": forecast_steps,
                },
                "metrics": metrics_dict,
                "components": {
                    "trend": [round(float(v), 4) for v in prediction["trend"]],
                    "dates": [str(d) for d in prediction["ds"]],
                },
            }
        except Exception as e:
            logger.error("Prophet fitting failed: %s", e, exc_info=True)
            return {"error": str(e), "model_type": "prophet"}

    # ----------------------------------------------------------------
    #  ML-Based Forecasting (lag features)
    # ----------------------------------------------------------------
    @staticmethod
    def fit_ml_forecast(
        series: pd.Series,
        forecast_steps: int = 30,
        model_type: str = "gradient_boosting",
        n_lags: int = 14,
        test_ratio: float = 0.2,
    ) -> Dict[str, Any]:
        """Use ML regressors with lag / calendar features for forecasting."""
        values = series.dropna().astype(float)
        start_time = time.time()

        # Build features
        feat_df = pd.DataFrame({"y": values})
        for lag in range(1, n_lags + 1):
            feat_df[f"lag_{lag}"] = feat_df["y"].shift(lag)

        # Calendar features (if datetime index)
        if isinstance(values.index, pd.DatetimeIndex):
            feat_df["dayofweek"] = values.index.dayofweek
            feat_df["month"] = values.index.month
            feat_df["day"] = values.index.day
            feat_df["quarter"] = values.index.quarter

        feat_df = feat_df.dropna()
        target = feat_df["y"]
        features = feat_df.drop(columns=["y"])

        # Train/test split (preserve temporal order)
        split_idx = int(len(features) * (1 - test_ratio))
        X_train, X_test = features.iloc[:split_idx], features.iloc[split_idx:]
        y_train, y_test = target.iloc[:split_idx], target.iloc[split_idx:]

        # Choose model
        models_map = {
            "gradient_boosting": GradientBoostingRegressor(n_estimators=200, max_depth=5, random_state=42),
            "random_forest": RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1),
            "linear": LinearRegression(),
        }
        model = models_map.get(model_type, models_map["gradient_boosting"])
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        metrics_dict = {
            "mae": round(float(mean_absolute_error(y_test, y_pred)), 4),
            "rmse": round(float(np.sqrt(mean_squared_error(y_test, y_pred))), 4),
            "r2": round(float(r2_score(y_test, y_pred)), 4),
        }
        try:
            metrics_dict["mape"] = round(float(mean_absolute_percentage_error(y_test, y_pred)), 4)
        except Exception:
            pass

        # Recursive forecast
        last_values = list(values.values[-n_lags:])
        forecast_values: list[float] = []
        last_date = values.index[-1]

        for step in range(forecast_steps):
            row: dict[str, float] = {}
            for lag in range(1, n_lags + 1):
                row[f"lag_{lag}"] = float(last_values[-lag])
            # Calendar features if available
            if isinstance(values.index, pd.DatetimeIndex):
                next_date = last_date + (values.index[-1] - values.index[-2]) * (step + 1)
                row["dayofweek"] = float(next_date.dayofweek)
                row["month"] = float(next_date.month)
                row["day"] = float(next_date.day)
                row["quarter"] = float(next_date.quarter)

            pred_val = float(model.predict(pd.DataFrame([row]))[0])
            forecast_values.append(round(pred_val, 4))
            last_values.append(pred_val)

        # Feature importance
        feat_imp: Optional[dict[str, float]] = None
        if hasattr(model, "feature_importances_"):
            imp = model.feature_importances_
            feat_imp = {name: round(float(v), 4) for name, v in zip(features.columns, imp)}
            feat_imp = dict(sorted(feat_imp.items(), key=lambda x: x[1], reverse=True))

        duration = round(time.time() - start_time, 2)

        return {
            "model_type": f"ml_{model_type}",
            "n_lags": n_lags,
            "training_time_sec": duration,
            "forecast": {
                "values": forecast_values,
                "steps": forecast_steps,
            },
            "metrics": metrics_dict,
            "feature_importance": feat_imp,
            "test_actual": [round(float(v), 4) for v in y_test],
            "test_predicted": [round(float(v), 4) for v in y_pred],
        }

    # ================================================================
    #  High-Level Convenience
    # ================================================================
    @classmethod
    def auto_forecast(
        cls,
        df: pd.DataFrame,
        datetime_col: str,
        value_col: str,
        forecast_steps: int = 30,
        models: Optional[List[str]] = None,
        freq: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Run multiple forecast models and return comparison.
        ``models`` can include: arima, exponential_smoothing, prophet, ml_gradient_boosting.
        """
        if models is None:
            models = ["arima", "exponential_smoothing", "ml_gradient_boosting"]
            if HAS_PROPHET:
                models.append("prophet")

        ts_df = cls.prepare_time_series(df, datetime_col, value_col, freq=freq)
        series = ts_df[value_col]

        results: Dict[str, Any] = {
            "datetime_col": datetime_col,
            "value_col": value_col,
            "data_points": len(series),
            "frequency": str(getattr(series.index, "freqstr", "unknown")),
            "summary": cls.ts_summary(series),
            "stationarity": cls.stationarity_test(series),
            "models": {},
        }

        for model_name in models:
            logger.info("Running forecast model: %s", model_name)
            try:
                if model_name == "arima":
                    results["models"]["arima"] = cls.fit_arima(series, forecast_steps=forecast_steps)
                elif model_name == "exponential_smoothing":
                    results["models"]["exponential_smoothing"] = cls.fit_exponential_smoothing(
                        series, forecast_steps=forecast_steps
                    )
                elif model_name == "prophet":
                    results["models"]["prophet"] = cls.fit_prophet(series, forecast_steps=forecast_steps)
                elif model_name.startswith("ml_"):
                    ml_type = model_name.replace("ml_", "")
                    results["models"][model_name] = cls.fit_ml_forecast(
                        series, forecast_steps=forecast_steps, model_type=ml_type
                    )
                else:
                    results["models"][model_name] = {"error": f"Unknown model: {model_name}"}
            except Exception as e:
                logger.error("Model %s failed: %s", model_name, e, exc_info=True)
                results["models"][model_name] = {"error": str(e)}

        # Find best model by MAE
        best_model = None
        best_mae = float("inf")
        for name, res in results["models"].items():
            if "metrics" in res and "mae" in res["metrics"]:
                mae = res["metrics"]["mae"]
                if mae < best_mae:
                    best_mae = mae
                    best_model = name
        results["best_model"] = best_model
        results["best_mae"] = round(best_mae, 4) if best_model else None

        return results

    @staticmethod
    def get_available_models() -> Dict[str, Any]:
        """Return which time-series models are available."""
        return {
            "arima": {"available": HAS_STATSMODELS, "description": "ARIMA / SARIMA — classical autoregressive model"},
            "exponential_smoothing": {"available": HAS_STATSMODELS, "description": "Holt-Winters Exponential Smoothing"},
            "prophet": {"available": HAS_PROPHET, "description": "Facebook Prophet — robust trend + seasonality"},
            "ml_gradient_boosting": {"available": True, "description": "Gradient Boosting with lag features"},
            "ml_random_forest": {"available": True, "description": "Random Forest with lag features"},
            "ml_linear": {"available": True, "description": "Linear Regression with lag features"},
        }
