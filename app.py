# app.py
import os
import math
import warnings
from datetime import datetime, timedelta, timezone
from flask import Flask
from flask_cors import CORS
from flask_pymongo import PyMongo
from pymongo import ASCENDING, UpdateOne, errors

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
#from sklearn.neural_network import MLPRegressor
from flask import request, jsonify
from time import perf_counter
from datetime import timezone
import pandas as pd
import numpy as np
import yfinance as yf


warnings.filterwarnings("ignore")

# ---------------------------------------------------------
# Flask & Mongo setup
# ---------------------------------------------------------
app = Flask(__name__, static_folder="templates")
CORS(app)

app.config["MONGO_URI"] = os.getenv(
    "MONGO_URI",
    "mongodb+srv://rajmukherjee1601:NO4ZmwVxiMrjP1Ym@cluster0.noyxs.mongodb.net/stockdb?retryWrites=true&w=majority&appName=Cluster0"
)
mongo = PyMongo(app)

# Ensure indexes
mongo.db.prices.create_index([("ticker", ASCENDING), ("dt", ASCENDING)], unique=True)
mongo.db.predictions.create_index([("ticker", ASCENDING), ("generated_at", ASCENDING)])

# ---------------------------------------------------------
# Audit log helper (pattern you use)
# ---------------------------------------------------------
def log_audit_action(db, page, action, data=None, user=None):
    audit_entry = {
        "page": page,
        "action": action,
        "data": data if data else {},
        "user": user or "system",
        "ts": datetime.now(timezone.utc),
    }
    db.Audit.insert_one(audit_entry)

# ---------------------------------------------------------
# Utils: yfinance chunked fetch (1h interval for ~1 year)
# ---------------------------------------------------------
def fetch_hourly_history_chunked(ticker: str,
                                 start: datetime,
                                 end: datetime,
                                 interval: str = "1h",
                                 chunk_days: int = 55) -> pd.DataFrame:
    """
    Fetches historical data in chunks to avoid yfinance period/interval caps.
    Returns UTC-indexed DataFrame with columns: [Open, High, Low, Close, Volume]
    """
    frames = []
    cur_start = start
    while cur_start < end:
        cur_end = min(cur_start + timedelta(days=chunk_days), end)
        df = yf.download(
            tickers=ticker,
            start=cur_start.strftime("%Y-%m-%d"),
            end=cur_end.strftime("%Y-%m-%d"),
            interval=interval,
            auto_adjust=False,
            prepost=False,
            threads=False,
            progress=False,
        )
        if df is not None and not df.empty:
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [c[0] for c in df.columns]
            df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
            if df.index.tz is None:
                df.index = df.index.tz_localize(timezone.utc)
            else:
                df.index = df.index.tz_convert(timezone.utc)
            frames.append(df)
        cur_start = cur_end

    if not frames:
        return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])

    out = pd.concat(frames)
    out = out[~out.index.duplicated(keep="last")]
    out.sort_index(inplace=True)
    return out

def upsert_history_to_db(ticker: str, df: pd.DataFrame) -> dict:
    """
    Bulk upsert hourly bars into MongoDB.
    """
    if df.empty:
        return {"matched": 0, "modified": 0, "upserted": 0}

    ops = []
    tk = ticker.upper()
    for ts, row in df.iterrows():
        doc = {
            "ticker": tk,
            "dt": ts,
            "open": float(row["Open"]),
            "high": float(row["High"]),
            "low": float(row["Low"]),
            "close": float(row["Close"]),
            "volume": float(row["Volume"]) if not math.isnan(row["Volume"]) else 0.0,
        }
        ops.append(
            UpdateOne(
                {"ticker": doc["ticker"], "dt": doc["dt"]},
                {"$set": doc},
                upsert=True,
            )
        )

    result = {"matched": 0, "modified": 0, "upserted": 0}
    try:
        bulk = mongo.db.prices.bulk_write(ops, ordered=False)
        result["matched"] = bulk.matched_count
        result["modified"] = bulk.modified_count
        result["upserted"] = len(bulk.upserted_ids) if bulk.upserted_ids else 0
    except errors.BulkWriteError as bwe:
        result["error"] = str(bwe.details)
    return result

# ---------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------
def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    roll_up = pd.Series(gain, index=series.index).rolling(period).mean()
    roll_down = pd.Series(loss, index=series.index).rolling(period).mean()
    rs = roll_up / (roll_down + 1e-9)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def build_features(price_df: pd.DataFrame):
    """
    Builds a supervised dataset to predict next-hour Close.
    Returns df, feature_cols
    """
    df = price_df.copy().sort_index()
    df["ret_1h"] = df["Close"].pct_change()
    df["lag_close_1"] = df["Close"].shift(1)
    df["lag_close_2"] = df["Close"].shift(2)
    df["lag_close_3"] = df["Close"].shift(3)
    df["sma_5"] = df["Close"].rolling(5).mean()
    df["sma_10"] = df["Close"].rolling(10).mean()
    df["ema_12"] = df["Close"].ewm(span=12, adjust=False).mean()
    df["ema_26"] = df["Close"].ewm(span=26, adjust=False).mean()
    df["macd"] = df["ema_12"] - df["ema_26"]
    df["vol_10"] = df["Close"].pct_change().rolling(10).std()
    df["rsi_14"] = compute_rsi(df["Close"], 14)
    df["vol_norm"] = (df["Volume"] + 1.0).apply(np.log)
    for col in ["sma_5", "sma_10", "ema_12", "ema_26"]:
        df[f"{col}_rel"] = df[col] / df["Close"]

    df["y_next_close"] = df["Close"].shift(-1)
    df = df.dropna()

    feature_cols = [
        "ret_1h",
        "lag_close_1", "lag_close_2", "lag_close_3",
        "sma_5_rel", "sma_10_rel", "ema_12_rel", "ema_26_rel",
        "macd", "vol_10", "rsi_14", "vol_norm",
    ]
    cols = feature_cols + ["y_next_close", "Close"]
    return df[cols], feature_cols

# ---------------------------------------------------------
# Modeling & Forecast
# ---------------------------------------------------------
def train_and_evaluate_models(dfX: pd.DataFrame, feature_cols: list):
    """
    Trains 5 models:
      1) Linear Regression
      2) Ridge
      3) Random Forest
      4) Gradient Boosting
      5) MLP (backprop)
    """
    X = dfX[feature_cols].values
    y = dfX["y_next_close"].values

    split_idx = int(len(X) * 0.8)
    X_train, y_train = X[:split_idx], y[:split_idx]
    X_test, y_test = X[split_idx:], y[split_idx:]

    models = {
        "1_LinearRegression": Pipeline([
            ("scaler", StandardScaler(with_mean=False)),
            ("model", LinearRegression())
        ]),
        "2_Ridge": Pipeline([
            ("scaler", StandardScaler(with_mean=False)),
            ("model", Ridge(alpha=1.0, random_state=42))
        ]),
        "3_RandomForest": RandomForestRegressor(
    n_estimators=80,      # was 350
    max_depth=8,         # cap depth
    random_state=42,
    n_jobs=1             # avoid lots of threads
),
"4_GradientBoosting": GradientBoostingRegressor(
    random_state=42,
    max_depth=3
),

        # "5_MLP_Backprop": Pipeline([
        #     ("scaler", StandardScaler()),
        #     ("model", MLPRegressor(
        #         hidden_layer_sizes=(64, 32),
        #         activation="relu",
        #         solver="adam",
        #         max_iter=500,
        #         random_state=42
        #     ))
        # ]),
    }

    results = {}
    for name, est in models.items():
        est.fit(X_train, y_train)
        y_pred = est.predict(X_test)
        mae = float(mean_absolute_error(y_test, y_pred))
        rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        mape = float(np.mean(np.abs((y_test - y_pred) / np.clip(np.abs(y_test), 1e-9, None))) * 100.0)
        r2 = float(r2_score(y_test, y_pred))
        results[name] = {"estimator": est, "metrics": {"MAE": mae, "RMSE": rmse, "MAPE_percent": mape, "R2": r2}}
    return results

def forecast_next_k_hours_from_present(model, base_price_df: pd.DataFrame, k: int, feature_cols: list):
    """
    Recursive 1-step-ahead predictions, k times, **anchored to current time**.
    Timestamps = (now at top-of-hour) + 1h, +2h, ..., +kh.
    """
    df = base_price_df.copy()
    preds = []

    # Anchor to the next three **future** top-of-hour timestamps from "now"
    now_utc = datetime.now(timezone.utc)
    anchor = now_utc.replace(minute=0, second=0, microsecond=0)

    last_close = df["Close"].iloc[-1]

    for step in range(1, k + 1):
        # Build features on the latest df
        dfX, _ = build_features(df)
        x_latest = dfX[feature_cols].iloc[-1].values.reshape(1, -1)
        next_close_pred = float(model.predict(x_latest)[0])

        # Timestamp for this forecast = now-top-of-hour + step hours
        ts = anchor + timedelta(hours=step)

        preds.append({
            "horizon_hours_ahead": step,
            "predicted_close": next_close_pred,
            "for_timestamp": ts.isoformat()
        })

        # Append a synthetic bar at 'ts' so next iteration has continuity
        # Use previous 'close' as open; simple OHLC heuristic
        new_row = df.iloc[-1][["Open", "High", "Low", "Close", "Volume"]].copy()
        new_row["Open"] = df["Close"].iloc[-1]
        new_row["Close"] = next_close_pred
        new_row["High"] = max(new_row["Open"], next_close_pred)
        new_row["Low"] = min(new_row["Open"], next_close_pred)

        # Ensure index is strictly increasing; if ts <= last index, push by +1h
        new_ts = ts
        if new_ts <= df.index[-1]:
            new_ts = df.index[-1] + timedelta(hours=1)

        df.loc[new_ts] = new_row

    return preds

# ---------------------------------------------------------
# Data access helpers
# ---------------------------------------------------------
def load_prices_from_db(ticker: str, min_rows: int = 500) -> pd.DataFrame:
    cursor = mongo.db.prices.find({"ticker": ticker.upper()}).sort("dt", ASCENDING)
    rows = list(cursor)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df.set_index("dt", inplace=True)
    df.index = pd.to_datetime(df.index, utc=True)
    df = df.rename(columns={
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close",
        "volume": "Volume"
    })[["Open", "High", "Low", "Close", "Volume"]].sort_index()
    return df

# ---------------------------------------------------------
# API Routes
# ---------------------------------------------------------
@app.route("/")
def serve_index():
    # Serve static dashboard (place index.html under ./static/)
    return app.send_static_file("index.html")

@app.route("/api")
def api_root():
    return jsonify({
        "ok": True,
        "message": "Stocks ML API is running.",
        "endpoints": {
            "/api/ingest?ticker=RELIANCE.NS": "Fetch & store last 2y hourly OHLCV",
            "/api/predict?ticker=RELIANCE.NS": "Train 5 models (5th=MLP backprop), metrics, forecast next 3 hours (from NOW)",
            "/api/predictions?ticker=RELIANCE.NS": "Get latest saved predictions for ticker"
        }
    })

@app.route("/api/ingest", methods=["GET"])
def api_ingest():
    ticker = (os.environ.get("DEFAULT_TICKER") or "").upper().strip()
    # allow query override
    from flask import request
    ticker = (request.args.get("ticker") or ticker or "").upper().strip()
    if not ticker:
        return jsonify({"ok": False, "error": "Missing ticker parameter"}), 400

    end = datetime.now(timezone.utc)
    start = end - timedelta(days=730)

    df = fetch_hourly_history_chunked(ticker, start, end, interval="1h", chunk_days=55)
    if df.empty:
        return jsonify({"ok": False, "ticker": ticker, "error": "No data returned from yfinance"}), 404

    up_res = upsert_history_to_db(ticker, df)
    log_audit_action(mongo.db, "prices", "ingest", {"ticker": ticker, "rows": int(len(df))})

    return jsonify({
        "ok": True,
        "ticker": ticker,
        "rows_in_chunked_frame": int(len(df)),
        "mongo_bulk_result": up_res
    })

@app.route("/api/predict", methods=["GET"])
def api_predict():
    from flask import request
    ticker = (request.args.get("ticker") or "").upper().strip()
    if not ticker:
        return jsonify({"ok": False, "error": "Missing ticker parameter"}), 400

    # Ensure data exists
    base_df = load_prices_from_db(ticker)
    if base_df.empty or len(base_df) < 200:
        end = datetime.now(timezone.utc)
        start = end - timedelta(days=730)
        df_new = fetch_hourly_history_chunked(ticker, start, end, interval="1h", chunk_days=55)
        if df_new.empty:
            return jsonify({"ok": False, "error": "No data available and ingest failed"}), 404
        upsert_history_to_db(ticker, df_new)
        base_df = load_prices_from_db(ticker)
        if base_df.empty or len(base_df) < 200:
            return jsonify({"ok": False, "error": "Insufficient data even after ingest"}), 422

    # Feature engineering
    supervised_df, feature_cols = build_features(base_df)
    if supervised_df.empty or supervised_df.shape[0] < 200:
        return jsonify({"ok": False, "error": "Not enough rows after feature engineering"}), 422

    # Train/evaluate
    results = train_and_evaluate_models(supervised_df, feature_cols)

    # Best by RMSE
    best_name = min(results.keys(), key=lambda k: results[k]["metrics"]["RMSE"])
    best_estimator = results[best_name]["estimator"]

    # === CHANGE: Forecast next 3 hours FROM PRESENT TIME ===
    preds = forecast_next_k_hours_from_present(
        best_estimator,
        base_df[["Open", "High", "Low", "Close", "Volume"]],
        k=3,
        feature_cols=feature_cols
    )

    metrics_out = {name: res["metrics"] for name, res in results.items()}

    # Save predictions
    doc = {
        "ticker": ticker,
        "generated_at": datetime.now(timezone.utc),
        "best_model": best_name,
        "all_metrics": metrics_out,
        "predictions": preds,
    }
    mongo.db.predictions.insert_one(doc)
    log_audit_action(mongo.db, "predictions", "create", {"ticker": ticker, "best_model": best_name})

    return jsonify({
        "ok": True,
        "ticker": ticker,
        "best_model": best_name,
        "metrics": metrics_out,
        "predictions_next_3_hours": preds
    })

@app.route("/api/predictions", methods=["GET"])
def api_predictions_latest():
    from flask import request
    ticker = (request.args.get("ticker") or "").upper().strip()
    if not ticker:
        return jsonify({"ok": False, "error": "Missing ticker parameter"}), 400

    doc = mongo.db.predictions.find_one({"ticker": ticker}, sort=[("generated_at", -1)])
    if not doc:
        return jsonify({"ok": False, "error": "No predictions found for ticker"}), 404

    doc["_id"] = str(doc["_id"])
    if isinstance(doc.get("generated_at"), datetime):
        doc["generated_at"] = doc["generated_at"].isoformat()

    return jsonify({"ok": True, "data": doc})

# ---------------------------------------------------------
# Entry
# ---------------------------------------------------------


@app.route("/rsi")
def serve_rsi_page():
    # make sure static/rsi.html exists (see step 2)
    return app.send_static_file("rsi.html")


# --- add these imports if missing ---


# ==== Add these imports at the top of app.py (if missing) ====


# ==== Your exact NSE list ====
NSE_TICKERS = [
    "RELIANCE.NS","TCS.NS","HDFCBANK.NS","ICICIBANK.NS","INFY.NS","ITC.NS","SBIN.NS",
    "BHARTIARTL.NS","HINDUNILVR.NS","LT.NS","AXISBANK.NS","KOTAKBANK.NS",
    "BAJFINANCE.NS","BAJAJFINSV.NS","ADANIENT.NS","ADANIPORTS.NS","MARUTI.NS",
    "SUNPHARMA.NS","ASIANPAINT.NS","M&M.NS","TATAMOTORS.NS","NTPC.NS","POWERGRID.NS",
    "ULTRACEMCO.NS","JSWSTEEL.NS","TATASTEEL.NS","NESTLEIND.NS","TITAN.NS","TECHM.NS",
    "WIPRO.NS","GRASIM.NS","CIPLA.NS","DIVISLAB.NS","DRREDDY.NS","BRITANNIA.NS",
    "EICHERMOT.NS","HEROMOTOCO.NS","BAJAJ-AUTO.NS","TATACONSUM.NS","HDFCLIFE.NS",
    "SBILIFE.NS","ICICIGI.NS","HINDALCO.NS","COALINDIA.NS","ONGC.NS","LTIM.NS",
    "APOLLOHOSP.NS","DMART.NS","DLF.NS","GODREJCP.NS","BSE.NS","UPL.NS","DIXON.NS",
    "SUZLON.NS","COFORGE.NS","BRIGADE.NS"
]

# ==== RSI (Wilder) utilities ====
def _rsi_series_wilder(close: pd.Series, period: int = 14) -> pd.Series:
    s = pd.Series(close, dtype="float64").dropna()
    if s.size < period + 1:
        return pd.Series(index=s.index, dtype="float64")
    delta = s.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def _last_iso(ts: pd.DatetimeIndex) -> str:
    if not len(ts):
        return ""
    t = ts[-1]
    if t.tz is None:
        t = t.tz_localize("UTC")
    else:
        t = t.tz_convert("UTC")
    return t.isoformat()

# ==== LIVE RSI API (NO DATABASE) ====
# Snapshot for ALL companies (default), optional series for ?ticker=SYM
@app.route("/api/rsi", methods=["GET"])
def rsi_live():
    """
    Live RSI calculated from Yahoo Finance (no DB).

    Modes:
      - Snapshot (default): present RSI for ALL NSE_TICKERS
      - Series (when ?ticker=SYM): RSI time-series for that symbol only

    Query params (optional):
      period=14              RSI period
      interval=1h            yfinance interval (e.g., 1h, 30m, 1d)
      yf_period=60d          yfinance period window (e.g., 60d, 6mo, 2y)
      series_points=1200     cap points for series mode
      order=desc|asc         sort snapshot by RSI (default desc)
    """
    t0 = perf_counter()

    # ---- inputs & sane defaults ----
    def _as_int(name, default, lo=None, hi=None):
        raw = request.args.get(name)
        try:
            x = int(raw) if raw is not None else int(default)
        except Exception:
            x = int(default)
        if lo is not None: x = max(lo, x)
        if hi is not None: x = min(hi, x)
        return x

    period = _as_int("period", 14, 2, 200)
    interval = (request.args.get("interval") or "1h").lower()
    yf_period = (request.args.get("yf_period") or ("60d" if interval.endswith(("m","h")) else "2y")).lower()
    series_points = _as_int("series_points", 1200, period + 1, 20000)
    order = (request.args.get("order") or "desc").lower()
    if order not in ("asc", "desc"):
        order = "desc"

    ticker = (request.args.get("ticker") or "").strip().upper()

    # ==== SERIES MODE: single ticker live from Yahoo ====
    if ticker:
        try:
            df = yf.download(
                tickers=[ticker], interval=interval, period=yf_period,
                auto_adjust=False, progress=False, threads=False, group_by="ticker"
            )
            # When single ticker, df has plain columns
            if isinstance(df.columns, pd.MultiIndex):
                # some builds still return MultiIndex for a single ticker
                df = df[ticker]

            close = df.get("Close")
            if close is None or close.dropna().size < period + 1:
                return jsonify(ok=False, error=f"Not enough data for {ticker} to compute RSI({period})"), 404

            rsi_series = _rsi_series_wilder(close, period).dropna()
            data = [{"timestamp": ts.isoformat() if (ts.tz is not None) else ts.tz_localize("UTC").isoformat(),
                     "rsi": float(val)} for ts, val in rsi_series.items()]

            payload = {
                "ok": True,
                "scope": "series",
                "ticker": ticker,
                "period": period,
                "interval": interval,
                "yf_period": yf_period,
                "last": {
                    "timestamp": _last_iso(close.index),
                    "close": float(close.dropna().iloc[-1]),
                    "rsi": float(rsi_series.iloc[-1]) if len(rsi_series) else None
                },
                "data": data
            }
            payload["elapsed_ms"] = int((perf_counter() - t0) * 1000)
            return jsonify(payload), 200
        except Exception as e:
            return jsonify(ok=False, error=f"{ticker}: {e}"), 500

    # ==== SNAPSHOT MODE: compute present RSI for ALL NSE_TICKERS ====
    try:
        # One multi-ticker call (fastest path). yfinance accepts a list.
        df_all = yf.download(
            tickers=NSE_TICKERS, interval=interval, period=yf_period,
            auto_adjust=False, progress=False, threads=False, group_by="ticker"
        )
    except Exception as e:
        return jsonify(ok=False, error=f"download failed: {e}"), 500

    rows = []

    # When multiple tickers are requested, yfinance returns a column MultiIndex:
    #   top level = ticker, second level = OHLCV columns.
    if isinstance(df_all.columns, pd.MultiIndex):
        for tk in NSE_TICKERS:
            if tk not in df_all.columns.get_level_values(0):
                continue
            try:
                sub = df_all[tk]
                close = sub.get("Close").dropna()
                if close.size < period + 1:
                    continue
                rsi = _rsi_series_wilder(close, period).dropna()
                if rsi.empty:
                    continue
                rows.append({
                    "ticker": tk,
                    "timestamp": _last_iso(close.index),
                    "close": float(close.iloc[-1]),
                    "rsi": float(rsi.iloc[-1]),
                    "extreme": bool((rsi.iloc[-1] < 35) or (rsi.iloc[-1] > 75))
                })
            except Exception:
                continue
    else:
        # Fallback: if yfinance returned a single-frame (rare when list len==1)
        close = df_all.get("Close")
        if close is not None and close.dropna().size >= period + 1:
            rsi = _rsi_series_wilder(close, period).dropna()
            if not rsi.empty:
                rows.append({
                    "ticker": NSE_TICKERS[0],
                    "timestamp": _last_iso(close.index),
                    "close": float(close.iloc[-1]),
                    "rsi": float(rsi.iloc[-1]),
                    "extreme": bool((rsi.iloc[-1] < 35) or (rsi.iloc[-1] > 75))
                })

    # sort by RSI (default: high → low)
    rows.sort(key=lambda x: x["rsi"], reverse=(order != "asc"))

    payload = {
        "ok": True,
        "scope": "snapshot_live",
        "period": period,
        "interval": interval,
        "yf_period": yf_period,
        "order": order,
        "count": len(rows),
        "data": rows
    }
    payload["elapsed_ms"] = int((perf_counter() - t0) * 1000)
    return jsonify(payload), 200


#<------------------------------Stock news ------------------------------------>


# --- add these ---
import html as _html
import urllib.parse as _uq
from email.utils import parsedate_to_datetime as _p2dt

try:
    import feedparser  # pip install feedparser
except Exception:
    feedparser = None


# ==== Google News helpers (24h sentinel) ====
_NEWS_CACHE = {}  # simple in-memory cache: {key: (ts, data)}

def _news_cache_get(key, ttl_sec=600):
    now = datetime.now(timezone.utc)
    v = _NEWS_CACHE.get(key)
    if not v:
        return None
    ts, data = v
    if (now - ts).total_seconds() > ttl_sec:
        return None
    return data

def _news_cache_set(key, data):
    _NEWS_CACHE[key] = (datetime.now(timezone.utc), data)

def _fetch_google_news(query: str, hours: int = 24, max_items: int = 20):
    """
    Google News RSS fetch for last <hours>, en-IN feed.
    Uses feedparser if available, otherwise falls back to stdlib XML parsing.
    """
    url = (
        "https://news.google.com/rss/search?q="
        + _uq.quote_plus(f"{query} when:{hours}h")
        + "&hl=en-IN&gl=IN&ceid=IN:en"
    )
    cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)

    # Primary path: feedparser
    if feedparser is not None:
        feed = feedparser.parse(url)
        out = []
        for e in getattr(feed, "entries", []):
            published = None
            if getattr(e, "published", None):
                try:
                    published = _p2dt(e.published)
                    if published.tzinfo is None:
                        published = published.replace(tzinfo=timezone.utc)
                    published = published.astimezone(timezone.utc)
                except Exception:
                    published = None
            if published and published < cutoff:
                continue
            src = None
            if hasattr(e, "source") and getattr(e.source, "title", None):
                src = e.source.title
            elif "source" in e and isinstance(e["source"], dict):
                src = e["source"].get("title")
            out.append({
                "title": _html.unescape(getattr(e, "title", "")),
                "link": getattr(e, "link", ""),
                "source": src,
                "published": published.isoformat() if published else None,
            })
            if len(out) >= max_items:
                break
        return out

    # Fallback path: stdlib XML
    import xml.etree.ElementTree as _ET
    from urllib.request import urlopen, Request
    try:
        req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
        raw = urlopen(req, timeout=10).read()
        root = _ET.fromstring(raw)
        out = []
        for item in root.findall(".//item"):
            title = (item.findtext("title") or "").strip()
            link = (item.findtext("link") or "").strip()
            pub = item.findtext("{*}pubDate") or item.findtext("pubDate")
            published = None
            if pub:
                try:
                    published = _p2dt(pub)
                    if published.tzinfo is None:
                        published = published.replace(tzinfo=timezone.utc)
                    published = published.astimezone(timezone.utc)
                except Exception:
                    published = None
            if published and published < cutoff:
                continue
            src = item.findtext("{*}source")
            out.append({
                "title": _html.unescape(title),
                "link": link,
                "source": src,
                "published": published.isoformat() if published else None,
            })
            if len(out) >= max_items:
                break
        return out
    except Exception:
        return []


# Simple rule-based sentiment (lightweight, no extra deps)
_POS = {"beat", "growth", "surge", "record", "strong", "upgrade", "relief", "bullish", "wins", "approves", "expands"}
_NEG = {"probe", "fraud", "ban", "crackdown", "fall", "slump", "cuts", "downgrade", "penalty", "losses", "delay"}

def _sentinel_sentiment(text: str):
    t = (text or "").lower()
    score = sum(w in t for w in _POS) - sum(w in t for w in _NEG)
    label = "Positive" if score > 0 else ("Negative" if score < 0 else "Neutral")
    return {"label": label, "score": score}

# Mapping: keyword → likely impacted tickers (keep it small & high-signal)
_IMPACT_RULES = [
    (r"\b(rbi|repo|rate hike|rate cut|inflation|cpi|mpc)\b", ["HDFCBANK.NS","ICICIBANK.NS","KOTAKBANK.NS","SBIN.NS","AXISBANK.NS"]),
    (r"\b(crude|brent|opec|oil price)\b", ["BPCL.NS","IOC.NS","HPCL.NS","ASIANPAINT.NS","BERGEPAINT.NS","INDIGO.NS"]),
    (r"\b(sebi|derivative curbs|f&o|ipo|buyback)\b", ["NSE-themed brokers (ANGELONE.NS, ISEC.NS)","Wider market"]),
    (r"\b(export|duty|import|gst|budget|pli)\b", ["TATAMOTORS.NS","MARUTI.NS","TATASTEEL.NS","JSWSTEEL.NS","SUNPHARMA.NS"]),
    (r"\b(power|renewable|solar|battery|pumped storage|bess)\b", ["ADANIGREEN.NS","TATAPOWER.NS","NTPC.NS","RELIANCE.NS"]),
]

import re as _re
def _likely_impacted_ticks(text: str):
    txt = (text or "")
    hits = []
    for patt, ticks in _IMPACT_RULES:
        if _re.search(patt, txt, flags=_re.I):
            hits.extend(ticks)
    # dedupe while preserving order
    seen, out = set(), []
    for t in hits:
        if t not in seen:
            seen.add(t); out.append(t)
    return out
# ==== Sentinel News: Market-moving India news (24h) ====
@app.route("/api/news/market", methods=["GET"])
def api_news_market():
    key = "market_24h"
    cached = _news_cache_get(key)
    if cached:
        return {"ok": True, **cached}

    query = "(RBI OR SEBI OR government OR policy OR budget OR crude oil OR OPEC OR inflation OR GDP OR IPO OR tax) India"
    items = _fetch_google_news(query, hours=24, max_items=28)
    for it in items:
        it["sentiment"] = _sentinel_sentiment(it.get("title", ""))
        it["likely_impacted"] = _likely_impacted_ticks((it.get("title","") + " " + (it.get("source") or "")))

    impacted_all = sorted({t for it in items for t in (it.get("likely_impacted") or [])})
    payload = {"query": query, "count": len(items), "impacted_all": impacted_all, "items": items}
    _news_cache_set(key, payload)
    return {"ok": True, **payload}

# ==== Sentinel News: Selected stock (24h) ====
@app.route("/api/news/stock", methods=["GET"])
def api_news_stock():
    from flask import request
    ticker = (request.args.get("ticker") or "").strip()
    if not ticker:
        return {"ok": False, "error": "ticker required"}, 400

    # turn "RELIANCE.NS" -> "RELIANCE" and add Indian context
    base = ticker.split(".")[0]
    query = f'"{base}" (stock OR shares OR NSE OR BSE) India'
    key = f"stock_24h::{base.upper()}"
    cached = _news_cache_get(key)
    if cached:
        return {"ok": True, "ticker": ticker, **cached}

    items = _fetch_google_news(query, hours=24, max_items=20)
    for it in items:
        it["sentiment"] = _sentinel_sentiment(it.get("title",""))

    payload = {"query": query, "count": len(items), "items": items}
    _news_cache_set(key, payload)
    return {"ok": True, "ticker": ticker, **payload}


@app.route("/live")
def serve_live_page():
    # make sure templates/live.html exists (see next section)
    return app.send_static_file("live.html")


@app.route("/api/live_price", methods=["GET"])
def api_live_price():
    """
    Return recent intraday prices (live-ish) for a given ticker from Yahoo Finance.

    Query params:
      ticker   = symbol like RELIANCE.NS (required)
      period   = yfinance period (default: 1d)
      interval = yfinance interval (default: 5m)

    Response:
      {
        ok: True/False,
        ticker: "...",
        last_price: float,
        last_timestamp: ISO string,
        data: [
          {timestamp, open, high, low, close, volume},
          ...
        ]
      }
    """
    from flask import request

    ticker = (request.args.get("ticker") or "").strip().upper()
    if not ticker:
        return jsonify({"ok": False, "error": "Missing ticker parameter"}), 400

    period = (request.args.get("period") or "1d").lower()
    interval = (request.args.get("interval") or "5m").lower()

    try:
        df = yf.download(
            tickers=[ticker],
            period=period,
            interval=interval,
            auto_adjust=False,
            progress=False,
            threads=False,
            group_by="ticker"
        )

        # When single ticker, df can be plain columns or MultiIndex, handle both
        if isinstance(df.columns, pd.MultiIndex):
            df = df[ticker]

        if df is None or df.empty:
            return jsonify({"ok": False, "error": "No price data returned"}), 404

        df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
        df.dropna(subset=["Close"], inplace=True)

        rows = []
        for ts, row in df.iterrows():
            if ts.tz is None:
                ts = ts.tz_localize(timezone.utc)
            else:
                ts = ts.tz_convert(timezone.utc)
            rows.append({
                "timestamp": ts.isoformat(),
                "open": float(row["Open"]),
                "high": float(row["High"]),
                "low": float(row["Low"]),
                "close": float(row["Close"]),
                "volume": float(row["Volume"]) if not math.isnan(row["Volume"]) else 0.0
            })

        if not rows:
            return jsonify({"ok": False, "error": "No usable rows in price data"}), 404

        last = rows[-1]
        return jsonify({
            "ok": True,
            "ticker": ticker,
            "period": period,
            "interval": interval,
            "last_price": last["close"],
            "last_timestamp": last["timestamp"],
            "data": rows
        })
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port , debug=False)
